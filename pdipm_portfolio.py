"""
pdipm_portfolio.py
==================
Solver primal-dual (predictor-corrector de Mehrotra) ESPECIALIZADO para

    min   w' Σ w  - gamma * mu' w
    s.t.  1' w = 1,   w >= 0,   ||w - w_prev||_1 <= L

con la misma formulación en x = [w ; u] que `build_portfolio_qp`
(u_i >= |w_i - w_prev_i|, 1'u <= L) y el mismo orden de filas de G:

    [ w - u <= w_prev ;  -w - u <= -w_prev ;  1'u <= L ;  -w <= 0 ]    (3N+1 filas)

Por qué existe
--------------
`PrimalDualQPSolver` (genérico) factoriza en cada iteración una matriz KKT densa
de tamaño (2N+1) y arma G' W G con G densa de (3N+1) x 2N. Con N = 500 son
sistemas de ~1000 y una G de 1501 x 1000: ~0.5-2 s por rebalanceo.

Aquí G' W G tiene estructura (W = diag(z/s) con bloques W1, W2, W3, W4):

    H_ww = 2κΣ + diag(W1 + W2 + W4)
    H_wu = diag(W2 - W1)                       =: diag(E)
    H_uu = diag(W1 + W2) + W3 · 11'            =: diag(D) + W3 · 11'

H_uu es diagonal + rango 1 => se invierte por Sherman-Morrison y u se elimina
(complemento de Schur). Queda un sistema N x N definido positivo

    S = 2κΣ + diag(4 W1 W2 / (W1 + W2) + W4) + γ v v',
    v = E / D,   γ = 1 / (1/W3 + Σ_i 1/D_i),

que se resuelve por Cholesky, más la restricción 1'w = 1 por un complemento de
Schur escalar. Costo por iteración: O(N^3/3) (Cholesky) en vez de O((2N)^3/3 + m n^2).
Se conserva el refinamiento iterativo contra el KKT SIN regularizar y la misma
política de regularización creciente del solver genérico.

Escalado de variables (importante para N grande)
------------------------------------------------
Los pesos son O(1/N). Con tolerancias ABSOLUTAS (tol=1e-8) sobre residuos y
duality gap, el solver declaraba "Optimal" demasiado pronto cuando N crece: en
pruebas contra CVXPY el objetivo quedaba 0.4 % (N=100), 10 % (N=250) y 45 %
(N=500) por encima del óptimo. Se resuelve en unidades de 1/N (x̂ = N·x: pesos y
holguras O(1), 1'ŵ = N, h → N·h) y se desescala al devolver. Entonces las
tolerancias actúan sobre cantidades O(1) y el óptimo coincide con CVXPY.
Nota: `PrimalDualQPSolver` (genérico) conserva la misma limitación.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from pdipm_qp_solver import QPInputError, QPNumericalError

FloatArray = NDArray[np.float64]


def _inf_norm(v: FloatArray) -> float:
    return float(np.max(np.abs(v))) if v.size else 0.0


def _max_step(v: FloatArray, dv: FloatArray) -> float:
    neg = dv < 0.0
    if not np.any(neg):
        return np.inf
    return float(np.min(-v[neg] / dv[neg]))


# --------------------------------------------------------------------------- #
# Operadores estructurados de G (sin materializar la matriz)
# --------------------------------------------------------------------------- #
def _G(w: FloatArray, u: FloatArray) -> FloatArray:
    """G x para x = [w; u]."""
    return np.concatenate([w - u, -w - u, [u.sum()], -w])


def _Gt(v: FloatArray, N: int) -> Tuple[FloatArray, FloatArray]:
    """G' v  separado en (parte w, parte u)."""
    v1, v2, v3, v4 = v[:N], v[N:2 * N], v[2 * N], v[2 * N + 1:]
    return v1 - v2 - v4, -v1 - v2 + v3


# --------------------------------------------------------------------------- #
# Sistema KKT reducido
# --------------------------------------------------------------------------- #
class _ReducedKKT:
    """Factorización del KKT  [[Q + G'WG, A'],[A, 0]]  con A = [1', 0]."""

    def __init__(self, Q_ww: FloatArray, W: FloatArray, N: int, reg: float) -> None:
        self.N = N
        self.Q_ww = Q_ww
        W1, W2, W3, W4 = W[:N], W[N:2 * N], float(W[2 * N]), W[2 * N + 1:]
        D = W1 + W2                                   # > 0 siempre (z, s > 0)
        self.D = D
        self.E = W2 - W1
        self.W3 = W3
        self.diag_ww = D + W4                         # parte diagonal de H_ww sin Q
        self.inv_D = 1.0 / D
        self.gamma = 1.0 / (1.0 / W3 + float(self.inv_D.sum()))
        d = 4.0 * W1 * W2 / D + W4                    # forma estable de D+W4-E²/D
        v = self.E * self.inv_D                       # acotado en [-1, 1]

        S = Q_ww + self.gamma * np.outer(v, v)
        idx = np.arange(N)
        S[idx, idx] += d + reg
        self._cho = cho_factor(S, lower=True, check_finite=False)  # LinAlgError si no es PD
        ones = np.ones(N)
        self._Sinv1 = cho_solve(self._cho, ones, check_finite=False)
        self._b = float(self._Sinv1.sum())
        if not (np.isfinite(self._b) and self._b > 0.0):
            raise LinAlgError("S^{-1} 1 no es finito/positivo.")
        self.norm_K = (float(np.linalg.norm(Q_ww, np.inf)) + float(self.diag_ww.max())
                       + float(np.abs(self.E).max()) + float(D.max()) + W3 * N + 1.0)

    def _hss_inv(self, y: FloatArray) -> FloatArray:
        q = y * self.inv_D
        return q - self.gamma * self.inv_D * float(q.sum())

    def solve(self, rw: FloatArray, ru: FloatArray, rho: float
              ) -> Tuple[FloatArray, FloatArray, float]:
        """Resuelve K [dw; du; dnu] = [rw; ru; rho]."""
        rt = rw - self.E * self._hss_inv(ru)
        sinv_rt = cho_solve(self._cho, rt, check_finite=False)
        dnu = (float(sinv_rt.sum()) - rho) / self._b
        dw = sinv_rt - self._Sinv1 * dnu
        du = self._hss_inv(ru - self.E * dw)
        return dw, du, dnu

    def matvec(self, dw: FloatArray, du: FloatArray, dnu: float
               ) -> Tuple[FloatArray, FloatArray, float]:
        """K [dw; du; dnu] SIN regularizar (para el refinamiento iterativo)."""
        gw = self.Q_ww @ dw + self.diag_ww * dw + self.E * du + dnu
        gu = self.E * dw + self.D * du + self.W3 * float(du.sum())
        return gw, gu, float(dw.sum())


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class PortfolioQPSolver:
    """Mehrotra predictor-corrector para mínima varianza long-only con turnover L1.

    Interfaz: ``solve(Sigma, w_prev, L, mu=None, gamma=0.0)`` devuelve el mismo
    diccionario que `PrimalDualQPSolver.fit` (con x_opt = [w; u]).
    """

    def __init__(self, tol: float = 1e-8, max_iter: int = 100,
                 reg_init: float = 1e-10, reg_max: float = 1e-5, reg_growth: float = 100.0,
                 resid_tol: float = 1e-10, resid_accept: float = 1e-6,
                 step_fraction: float = 0.99, max_backtrack: int = 40,
                 refine_steps: int = 3, scale_cost: bool = True,
                 divergence_limit: float = 1e14, verbose: bool = False) -> None:
        if not (0.0 < step_fraction < 1.0):
            raise QPInputError("step_fraction debe estar en (0, 1).")
        if tol <= 0 or max_iter < 1:
            raise QPInputError("tol debe ser > 0 y max_iter >= 1.")
        self.tol, self.max_iter = tol, max_iter
        self.reg_init, self.reg_max, self.reg_growth = reg_init, reg_max, reg_growth
        self.resid_tol, self.resid_accept = resid_tol, resid_accept
        self.step_fraction, self.max_backtrack = step_fraction, max_backtrack
        self.refine_steps, self.scale_cost = refine_steps, scale_cost
        self.divergence_limit, self.verbose = divergence_limit, verbose
        self._reg = reg_init

    # ---------------------------- álgebra lineal --------------------------- #
    def _refined(self, fac: _ReducedKKT, rw: FloatArray, ru: FloatArray, rho: float
                 ) -> Tuple[Tuple[FloatArray, FloatArray, float], float]:
        tiny = np.finfo(np.float64).tiny
        rhs_norm = max(_inf_norm(rw), _inf_norm(ru), abs(rho))

        def resid(sol: Tuple[FloatArray, FloatArray, float]
                  ) -> Tuple[Tuple[FloatArray, FloatArray, float], float]:
            gw, gu, ge = fac.matvec(*sol)
            r = (rw - gw, ru - gu, rho - ge)
            sol_norm = max(_inf_norm(sol[0]), _inf_norm(sol[1]), abs(sol[2]))
            be = max(_inf_norm(r[0]), _inf_norm(r[1]), abs(r[2])) / (
                fac.norm_K * sol_norm + rhs_norm + tiny)
            return r, be

        sol = fac.solve(rw, ru, rho)
        r, be = resid(sol)
        for _ in range(self.refine_steps):
            if not np.isfinite(be) or be < 1e-15:
                break
            corr = fac.solve(*r)
            cand = (sol[0] + corr[0], sol[1] + corr[1], sol[2] + corr[2])
            r2, be2 = resid(cand)
            if not (be2 < be):
                break
            sol, r, be = cand, r2, be2
        return sol, float(be)

    def _factor_and_solve(self, Q_ww: FloatArray, W: FloatArray, N: int,
                          rw: FloatArray, ru: FloatArray, rho: float, reg_start: float
                          ) -> Tuple[_ReducedKKT, Tuple[FloatArray, FloatArray, float], float]:
        reg = max(reg_start, self.reg_init)
        last_exc: Optional[BaseException] = None
        while True:
            try:
                fac = _ReducedKKT(Q_ww, W, N, reg)
                sol, be = self._refined(fac, rw, ru, rho)
                finite = all(np.all(np.isfinite(p)) for p in sol[:2]) and np.isfinite(sol[2])
                if finite and np.isfinite(be):
                    if be <= self.resid_tol or (reg >= self.reg_max and be <= self.resid_accept):
                        return fac, sol, reg
                    last_exc = None
                else:
                    last_exc = LinAlgError("solución no finita")
            except (LinAlgError, ValueError, FloatingPointError) as exc:
                last_exc = exc
            if reg >= self.reg_max:
                raise QPNumericalError("Sistema KKT reducido singular/mal condicionado "
                                       f"incluso con regularización {self.reg_max:.1e}.") from last_exc
            reg = min(reg * self.reg_growth, self.reg_max)

    # ------------------------------- API ----------------------------------- #
    def solve(self, Sigma: FloatArray, w_prev: FloatArray, L: float,
              mu: Optional[FloatArray] = None, gamma: float = 0.0) -> Dict[str, Any]:
        Sigma = np.asarray(Sigma, dtype=np.float64)
        w_prev = np.asarray(w_prev, dtype=np.float64).ravel()
        N = w_prev.size
        if Sigma.shape != (N, N):
            raise QPInputError(f"Sigma debe ser ({N},{N}); recibido {Sigma.shape}.")
        if not (np.all(np.isfinite(Sigma)) and np.all(np.isfinite(w_prev))):
            raise QPInputError("Sigma o w_prev contienen NaN/Inf.")
        if not (np.isfinite(L) and L > 0.0):
            raise QPInputError("El presupuesto de turnover L debe ser finito y > 0.")
        if gamma != 0.0 and mu is None:
            raise QPInputError("Se requiere `mu` si gamma != 0.")
        c_w0 = np.zeros(N) if (mu is None or gamma == 0.0) else -gamma * np.asarray(mu, dtype=np.float64).ravel()
        if c_w0.size != N:
            raise QPInputError(f"mu debe tener {N} elementos.")

        # Variables en unidades de 1/N: x̂ = N x. El objetivo se multiplica por N² (constante
        # positiva): ŵ'Σŵ + N c'ŵ.  Luego κ normaliza la escala del costo como antes.
        vs = float(N)
        Q0 = 2.0 * Sigma
        c_hat = vs * c_w0
        kappa = 1.0
        if self.scale_cost:
            nrm = max(float(np.linalg.norm(Q0, np.inf)), _inf_norm(c_hat))
            if nrm > 0.0:
                kappa = 1.0 / nrm
        Qw, cw = kappa * Q0, kappa * c_hat
        h = vs * np.concatenate([w_prev, -w_prev, [L], np.zeros(N)])
        m = 3 * N + 1
        self._reg = self.reg_init

        hist: Dict[str, List[float]] = {"primal": [], "dual": [], "gap": []}
        status, message = "Max_Iter_Reached", "Se alcanzó max_iter sin converger."

        def result(w: FloatArray, u: FloatArray, nu: float, z: FloatArray, s: FloatArray,
                   status: str, iters: int, message: str) -> Dict[str, Any]:
            w, u = w / vs, u / vs                       # desescalado a pesos reales
            obj = float(w @ Sigma @ w + c_w0 @ w) if np.all(np.isfinite(w)) else float("nan")
            return {"x_opt": np.concatenate([w, u]), "status": status, "iterations": iters,
                    "kkt_residuals": hist,              # en unidades escaladas (1/N)
                    "objective": obj, "nu_opt": np.array([nu / (kappa * vs)]),
                    "z_opt": z / (kappa * vs), "slack_opt": s / vs, "message": message}

        # ---- Punto inicial (heurística de Mehrotra; mismo esquema que el genérico) ----
        try:
            ghw, ghu = _Gt(h, N)
            _, (w, u, nu), _ = self._factor_and_solve(Qw, np.ones(m), N, -cw + ghw, ghu, vs,
                                                      self.reg_init)
        except QPNumericalError as exc:
            nan = np.full(N, np.nan)
            return result(nan, nan, float("nan"), np.zeros(m), np.zeros(m), "Numerical_Error", 0,
                          f"Fallo en el punto inicial: {exc}")
        s0 = h - _G(w, u)
        z0 = -s0
        s_hat = s0 + max(-1.5 * s0.min(), 0.0)
        z_hat = z0 + max(-1.5 * z0.min(), 0.0)
        sz = float(s_hat @ z_hat)
        if sz > 0.0 and z_hat.sum() > 0.0 and s_hat.sum() > 0.0:
            s = s_hat + 0.5 * sz / z_hat.sum()
            z = z_hat + 0.5 * sz / s_hat.sum()
        else:
            s, z = np.ones(m), np.ones(m)

        if self.verbose:
            print(f"{'it':>3} {'primal':>10} {'dual':>10} {'gap':>10} {'alpha':>8} {'sigma':>8}")

        k, alpha = 0, float("nan")
        for k in range(self.max_iter + 1):
            Qw_w = Qw @ w
            gzw, gzu = _Gt(z, N)
            r_dw = Qw_w + cw + nu + gzw
            r_du = gzu
            r_eq = float(w.sum()) - vs
            r_in = _G(w, u) + s - h
            pres = max(abs(r_eq), _inf_norm(r_in))
            dres = max(_inf_norm(r_dw), _inf_norm(r_du))
            gap = float(s @ z) / m
            hist["primal"].append(pres); hist["dual"].append(dres); hist["gap"].append(gap)
            if self.verbose:
                print(f"{k:3d} {pres:10.3e} {dres:10.3e} {gap:10.3e} {alpha:8.2e}")

            if not (np.isfinite(pres) and np.isfinite(dres) and np.isfinite(gap)):
                status, message = "Numerical_Error", "Residuos no finitos."
                break
            if max(_inf_norm(w), _inf_norm(u), abs(nu), _inf_norm(z)) > self.divergence_limit:
                status, message = ("Numerical_Error",
                                   "Divergencia de iterados: posible problema infactible o no acotado.")
                break
            if pres < self.tol and dres < self.tol and gap < self.tol:
                status, message = "Optimal", "Convergencia alcanzada."
                break
            if k == self.max_iter:
                break

            try:
                w, u, nu, s, z, alpha = self._step(Qw, N, w, u, nu, s, z,
                                                   r_dw, r_du, r_eq, r_in, gap)
            except QPNumericalError as exc:
                status, message = "Numerical_Error", str(exc)
                break
            if alpha < 1e-12:
                status, message = "Numerical_Error", "Paso de longitud ~0 (estancamiento)."
                break

        return result(w, u, nu, z, s, status, k, message)

    # ------------------------- paso de Mehrotra ----------------------------- #
    def _step(self, Qw: FloatArray, N: int, w: FloatArray, u: FloatArray, nu: float,
              s: FloatArray, z: FloatArray, r_dw: FloatArray, r_du: FloatArray,
              r_eq: float, r_in: FloatArray, gap: float
              ) -> Tuple[FloatArray, FloatArray, float, FloatArray, FloatArray, float]:
        W = z / s

        def rhs(r_c: FloatArray) -> Tuple[FloatArray, FloatArray, float]:
            t = (z * r_in - r_c) / s
            gtw, gtu = _Gt(t, N)
            return -r_dw - gtw, -r_du - gtu, -r_eq

        def split(sol: Tuple[FloatArray, FloatArray, float], r_c: FloatArray
                  ) -> Tuple[FloatArray, FloatArray, float, FloatArray, FloatArray]:
            dw, du, dnu = sol
            ds = -r_in - _G(dw, du)
            dz = (-r_c - z * ds) / s
            return dw, du, dnu, ds, dz

        # Predictor (affine scaling)
        r_c_aff = s * z
        fac, sol, reg_used = self._factor_and_solve(Qw, W, N, *rhs(r_c_aff), self._reg)
        self._reg = max(self.reg_init, reg_used / self.reg_growth)
        dw, du, dnu, ds, dz = split(sol, r_c_aff)

        alpha_aff = min(1.0, _max_step(s, ds), _max_step(z, dz))
        mu_aff = float((s + alpha_aff * ds) @ (z + alpha_aff * dz)) / s.size
        sigma = float(np.clip((mu_aff / gap) ** 3, 0.0, 1.0)) if gap > 0.0 else 0.0

        # Corrector
        r_c = s * z + ds * dz - sigma * gap
        rw, ru, rho = rhs(r_c)
        sol_c, be = self._refined(fac, rw, ru, rho)
        ok = (np.isfinite(be) and be <= self.resid_tol
              and all(np.all(np.isfinite(p)) for p in sol_c[:2]) and np.isfinite(sol_c[2]))
        if not ok:
            fac, sol_c, reg_used = self._factor_and_solve(
                Qw, W, N, rw, ru, rho, min(reg_used * self.reg_growth, self.reg_max))
            self._reg = max(self.reg_init, reg_used / self.reg_growth)
        dw, du, dnu, ds, dz = split(sol_c, r_c)

        # Longitud de paso
        alpha_max = min(_max_step(s, ds), _max_step(z, dz))
        alpha = min(1.0, self.step_fraction * alpha_max)
        for _ in range(self.max_backtrack):
            s_new, z_new = s + alpha * ds, z + alpha * dz
            if np.all(s_new > 0.0) and np.all(z_new > 0.0):
                break
            alpha *= 0.5
        else:
            raise QPNumericalError("Backtracking falló: no se pudo mantener s > 0 y z > 0.")
        return w + alpha * dw, u + alpha * du, nu + alpha * dnu, s_new, z_new, float(alpha)
