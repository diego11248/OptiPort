"""
pdipm_qp_solver.py
==================

Solver de Programación Cuadrática (QP) convexa por Punto Interior Primal-Dual
con esquema Predictor-Corrector de Mehrotra, implementado solo con NumPy/SciPy
(factorizaciones y álgebra lineal densa; sin CVXPY, ECOS, OSQP ni scipy.optimize).

Problema resuelto
-----------------
    minimizar     (1/2) x' Q x + c' x
    sujeto a      A_eq  x  = b_eq
                  A_ineq x <= b_ineq            (Q simétrica semidefinida positiva)

Formulación interna (G := A_ineq, h := b_ineq, A := A_eq)
---------------------------------------------------------
Se introducen holguras s > 0 y multiplicadores z > 0:

    G x + s = h,    s > 0,    z > 0

Condiciones KKT (Lagrangiano  L = 1/2 x'Qx + c'x + nu'(Ax-b) + z'(Gx+s-h)):

    r_dual = Q x + c + A' nu + G' z   = 0
    r_eq   = A x - b                  = 0
    r_in   = G x + s - h              = 0
    r_cmp  = s ∘ z                    = 0     (complementariedad)

Sistema de Newton reducido (eliminando ds y dz, con W = diag(z / s)):

    [ Q + G' W G    A' ] [ dx  ]   [ rhs_dual   ]
    [     A          0 ] [ dnu ] = [ rhs_primal ]

    rhs_dual   = -r_dual - G' ( (z ∘ r_in - r_c) / s )
    rhs_primal = -r_eq
    ds         = -r_in - G dx
    dz         = ( -r_c - z ∘ ds ) / s

donde r_c = s∘z (predictor) o r_c = s∘z + ds_aff∘dz_aff - sigma*mu*1 (corrector).
Es un sistema simétrico indefinido (tipo punto de silla) que se factoriza con
LDL' (Bunch-Kaufman, scipy.linalg.ldl). La MISMA factorización se reutiliza
para predictor y corrector.

Notas de diseño
---------------
* Se usa un único paso alpha para primal y dual (en QP los pasos primal/dual
  están acoplados por Q; usar pasos distintos rompe la linealidad de r_dual).
* Regularización Tikhonov/proximal "quasi-definida": +eps*I en el bloque
  primal y -eps*I en el bloque dual. eps se adapta (x100) solo cuando la
  factorización falla o el error hacia atrás del solve queda por encima de
  `resid_tol`. Luego se aplica refinamiento iterativo contra la matriz KKT
  SIN regularizar, de modo que eps no sesga la solución.
* "Duality gap" = s'z / m con m = nº de desigualdades (en el enunciado "N").
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, ldl, solve_banded, solve_triangular
from scipy.sparse.linalg import LinearOperator, onenormest

FloatArray = NDArray[np.float64]

__all__ = [
    "PrimalDualQPSolver",
    "build_portfolio_qp",
    "split_solution",
    "QPSolverError",
    "QPInputError",
    "QPNumericalError",
]


# --------------------------------------------------------------------------- #
# Excepciones
# --------------------------------------------------------------------------- #
class QPSolverError(Exception):
    """Clase base de errores del solver."""


class QPInputError(QPSolverError, ValueError):
    """Datos de entrada inválidos (dimensiones, NaN/Inf, Q no convexa...)."""


class QPNumericalError(QPSolverError, ArithmeticError):
    """Fallo algebraico/numérico: factorización singular, residuo inaceptable,
    pérdida de positividad de (s, z), divergencia, etc."""


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _inf_norm(v: FloatArray) -> float:
    """Norma infinito segura para vectores vacíos."""
    return float(np.max(np.abs(v))) if v.size else 0.0


def _max_step(v: FloatArray, dv: FloatArray) -> float:
    """Mayor alpha >= 0 tal que v + alpha*dv >= 0 (test de razón, vectorizado).

    Devuelve +inf si ninguna componente de dv es negativa.
    """
    neg = dv < 0.0
    if not np.any(neg):
        return np.inf
    return float(np.min(-v[neg] / dv[neg]))


class _LDLFactor:
    """Factorización LDL' (Bunch-Kaufman) de una matriz simétrica indefinida.

    scipy.linalg.ldl devuelve (lu, d, perm) con  K = lu @ d @ lu.T  y
    lu[perm] triangular inferior unitaria. Entonces, con L = lu[perm]:

        K[perm][:, perm] = L @ D @ L.T

    D es diagonal por bloques (1x1 y 2x2) => tridiagonal simétrica, que se
    resuelve vectorizadamente con `solve_banded`.
    """

    def __init__(self, K: FloatArray) -> None:
        lu, d, perm = ldl(K, lower=True)  # check_finite=True: lanza ValueError si hay NaN/Inf
        if not (np.all(np.isfinite(lu)) and np.all(np.isfinite(d))):
            raise LinAlgError("LDL produjo factores no finitos.")
        n = d.shape[0]
        self._L: FloatArray = lu[perm]
        self._perm: NDArray[np.intp] = np.asarray(perm)
        ab = np.zeros((3, n))
        ab[0, 1:] = np.diag(d, 1)
        ab[1, :] = np.diag(d)
        ab[2, :-1] = np.diag(d, -1)
        self._ab = ab

    def solve(self, b: FloatArray) -> FloatArray:
        y = solve_triangular(self._L, b[self._perm], lower=True,
                             unit_diagonal=True, check_finite=False)
        z = solve_banded((1, 1), self._ab, y, check_finite=False)
        w = solve_triangular(self._L, z, lower=True, trans="T",
                             unit_diagonal=True, check_finite=False)
        x = np.empty_like(w)
        x[self._perm] = w
        return x


class _InverseOperator(LinearOperator):
    """Operador lineal K^{-1} basado en una factorización LDL'.

    Se define por subclase (en lugar de pasar `matvec=`/`rmatmat=` al
    constructor) para que funcione en cualquier versión de SciPy: el argumento
    `rmatmat` solo existe desde SciPy 1.4. K es simétrica, así que el adjunto
    coincide con el operador mismo.
    """

    def __init__(self, fac: _LDLFactor, n: int) -> None:
        super().__init__(dtype=np.float64, shape=(n, n))
        self._fac = fac

    def _matvec(self, v: FloatArray) -> FloatArray:
        return self._fac.solve(np.ravel(np.asarray(v, dtype=np.float64)))

    def _rmatvec(self, v: FloatArray) -> FloatArray:
        return self._matvec(v)

    def _matmat(self, V: FloatArray) -> FloatArray:
        V = np.asarray(V, dtype=np.float64)
        return np.column_stack([self._fac.solve(V[:, j]) for j in range(V.shape[1])])

    def _rmatmat(self, V: FloatArray) -> FloatArray:
        return self._matmat(V)


def _cond1_estimate(K: FloatArray, fac: _LDLFactor) -> float:
    """Estimador (Higham/Hager) de cond_1(K) = ||K||_1 * ||K^{-1}||_1.

    Usa `onenormest` sobre un operador que aplica K^{-1} vía LDL'; cuesta solo
    unas pocas resoluciones triangulares. Solo es informativo: ante cualquier
    fallo devuelve NaN en lugar de interrumpir el solver.
    """
    try:
        op = _InverseOperator(fac, K.shape[0])
        return float(np.linalg.norm(K, 1) * onenormest(op))
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class PrimalDualQPSolver:
    """QP convexo por Punto Interior Primal-Dual + Predictor-Corrector de Mehrotra.

    Parameters
    ----------
    tol : float
        Tolerancia para ||r_primal||_inf, ||r_dual||_inf y duality gap.
    max_iter : int
        Máximo de iteraciones de Newton.
    reg_init, reg_max, reg_growth : float
        Regularización Tikhonov inicial, máxima y factor de crecimiento.
    resid_tol : float
        Error hacia atrás (relativo) aceptado tras el refinamiento; por encima
        de este valor se aumenta la regularización y se refactoriza.
    resid_accept : float
        Error hacia atrás máximo aceptado cuando ya se llegó a `reg_max`.
    step_fraction : float
        Fracción al borde ("fraction-to-boundary") en (0, 1), típicamente 0.99.
    max_backtrack : int
        Máximo de reducciones por backtracking (alpha <- alpha/2) para
        garantizar s > 0 y z > 0 estrictos.
    refine_steps : int
        Pasos de refinamiento iterativo por cada resolución.
    relative_tol : bool
        Si True, escala las tolerancias por la magnitud de los datos.
    check_psd : bool
        Verifica que Q sea semidefinida positiva (eigvalsh, O(n^3) una vez).
    scale_cost : bool
        Si True, resuelve internamente con (Q, c) multiplicados por
        kappa = 1/max(||Q||_inf, ||c||_inf). Imprescindible en finanzas, donde
        Sigma ~ 1e-5 haría que una tolerancia absoluta fuese demasiado laxa.
        x_opt, nu_opt, z_opt y objective se devuelven en unidades originales;
        el historial de residuos dual/gap está en unidades del problema escalado.
    estimate_cond : bool
        Si True, estima cond_1 de la matriz KKT en cada iteración (diagnóstico).
    divergence_limit : float
        Si ||x||_inf, ||z||_inf o ||nu||_inf lo superan => 'Numerical_Error'
        (posible problema infactible/no acotado).
    verbose : bool
        Imprime una línea por iteración.
    """

    def __init__(
        self,
        tol: float = 1e-10,
        max_iter: int = 100,
        reg_init: float = 1e-10,
        reg_max: float = 1e-5,
        reg_growth: float = 100.0,
        resid_tol: float = 1e-10,
        resid_accept: float = 1e-6,
        step_fraction: float = 0.99,
        max_backtrack: int = 40,
        refine_steps: int = 3,
        relative_tol: bool = False,
        check_psd: bool = True,
        scale_cost: bool = True,
        estimate_cond: bool = False,
        divergence_limit: float = 1e14,
        verbose: bool = False,
    ) -> None:
        if not (0.0 < step_fraction < 1.0):
            raise QPInputError("step_fraction debe estar en (0, 1).")
        if tol <= 0 or max_iter < 1:
            raise QPInputError("tol debe ser > 0 y max_iter >= 1.")
        self.tol = tol
        self.max_iter = max_iter
        self.reg_init = reg_init
        self.reg_max = reg_max
        self.reg_growth = reg_growth
        self.resid_tol = resid_tol
        self.resid_accept = resid_accept
        self.step_fraction = step_fraction
        self.max_backtrack = max_backtrack
        self.refine_steps = refine_steps
        self.relative_tol = relative_tol
        self.check_psd = check_psd
        self.scale_cost = scale_cost
        self.estimate_cond = estimate_cond
        self.divergence_limit = divergence_limit
        self.verbose = verbose

        # Estado interno por llamada a fit()
        self._reg: float = reg_init
        self._reg_signs: FloatArray = np.empty(0)
        self._norm_K: float = 1.0
        self._last_cond: float = float("nan")

    # ------------------------------------------------------------------ #
    # Validación de datos
    # ------------------------------------------------------------------ #
    def _validate(
        self,
        Q: FloatArray,
        c: FloatArray,
        A_eq: Optional[FloatArray],
        b_eq: Optional[FloatArray],
        A_ineq: Optional[FloatArray],
        b_ineq: Optional[FloatArray],
    ) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        c = np.asarray(c, dtype=np.float64).ravel()
        n = c.size
        if n == 0:
            raise QPInputError("c no puede ser vacío.")
        Q = np.asarray(Q, dtype=np.float64)
        if Q.shape != (n, n):
            raise QPInputError(f"Q debe ser ({n},{n}); recibido {Q.shape}.")

        def _mat(M: Optional[FloatArray], v: Optional[FloatArray], name: str
                 ) -> Tuple[FloatArray, FloatArray]:
            if M is None or np.size(M) == 0:
                M_ = np.zeros((0, n))
            else:
                M_ = np.atleast_2d(np.asarray(M, dtype=np.float64))
            v_ = np.zeros(0) if v is None else np.asarray(v, dtype=np.float64).ravel()
            if M_.shape[1] != n:
                raise QPInputError(f"A_{name} debe tener {n} columnas; tiene {M_.shape[1]}.")
            if M_.shape[0] != v_.size:
                raise QPInputError(f"A_{name} ({M_.shape[0]} filas) y b_{name} ({v_.size}) no concuerdan.")
            return M_, v_

        A, b = _mat(A_eq, b_eq, "eq")
        G, h = _mat(A_ineq, b_ineq, "ineq")

        for name, arr in (("Q", Q), ("c", c), ("A_eq", A), ("b_eq", b), ("A_ineq", G), ("b_ineq", h)):
            if not np.all(np.isfinite(arr)):
                raise QPInputError(f"{name} contiene NaN o Inf.")

        asym = np.linalg.norm(Q - Q.T, np.inf) / max(1.0, np.linalg.norm(Q, np.inf))
        if asym > 1e-8:
            warnings.warn(f"Q no es simétrica (asimetría relativa {asym:.2e}); se simetriza.")
        Q = 0.5 * (Q + Q.T)

        if self.check_psd:
            try:
                lam = np.linalg.eigvalsh(Q)
            except np.linalg.LinAlgError as exc:
                raise QPInputError("No se pudo calcular el espectro de Q.") from exc
            if lam.min() < -1e-8 * max(1.0, np.abs(lam).max()):
                raise QPInputError(
                    f"Q no es semidefinida positiva (autovalor mínimo {lam.min():.3e}); "
                    "el problema no es convexo.")
        return Q, c, A, b, G, h

    # ------------------------------------------------------------------ #
    # Álgebra lineal robusta
    # ------------------------------------------------------------------ #
    def _refined_solve(self, K: FloatArray, fac: _LDLFactor, rhs: FloatArray
                       ) -> Tuple[FloatArray, float]:
        """Resuelve K sol = rhs con la factorización `fac` (de K regularizada) y
        refinamiento iterativo contra K sin regularizar.

        Returns
        -------
        sol, backward_error  con  be = ||r||_inf / (||K|| ||sol|| + ||rhs||).
        """
        tiny = np.finfo(np.float64).tiny

        def _be(sol: FloatArray) -> Tuple[FloatArray, float]:
            r = rhs - K @ sol
            den = self._norm_K * _inf_norm(sol) + _inf_norm(rhs) + tiny
            return r, _inf_norm(r) / den

        sol = fac.solve(rhs)
        r, be = _be(sol)
        for _ in range(self.refine_steps):
            if not np.isfinite(be) or be < 1e-15:
                break
            cand = sol + fac.solve(r)
            r2, be2 = _be(cand)
            if not (be2 < be):
                break
            sol, r, be = cand, r2, be2
        return sol, float(be)

    def _factor_and_solve(self, K: FloatArray, rhs: FloatArray, reg_start: float
                          ) -> Tuple[_LDLFactor, FloatArray, float]:
        """Factoriza K + diag(+eps, -eps) y resuelve; si falla o el error hacia
        atrás es alto, multiplica eps por `reg_growth` y reintenta."""
        reg = max(reg_start, self.reg_init)
        idx = np.arange(K.shape[0])
        last_exc: Optional[BaseException] = None
        while True:
            try:
                Kr = K.copy()
                Kr[idx, idx] += self._reg_signs * reg
                fac = _LDLFactor(Kr)
                sol, be = self._refined_solve(K, fac, rhs)
                if np.isfinite(be) and np.all(np.isfinite(sol)):
                    if be <= self.resid_tol or (reg >= self.reg_max and be <= self.resid_accept):
                        if self.estimate_cond:
                            self._last_cond = _cond1_estimate(Kr, fac)
                        return fac, sol, reg
                    last_exc = None
                else:
                    last_exc = LinAlgError("solución no finita")
            except (LinAlgError, ValueError, FloatingPointError) as exc:
                last_exc = exc
            if reg >= self.reg_max:
                raise QPNumericalError(
                    "Sistema KKT singular/mal condicionado incluso con la "
                    f"regularización máxima ({self.reg_max:.1e}).") from last_exc
            reg = min(reg * self.reg_growth, self.reg_max)

    # ------------------------------------------------------------------ #
    # Punto inicial (heurística de Mehrotra)
    # ------------------------------------------------------------------ #
    def _initial_point(
        self, Q: FloatArray, c: FloatArray, A: FloatArray, b: FloatArray,
        G: FloatArray, h: FloatArray,
    ) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """x0, nu0 de min 1/2 x'Qx + c'x + 1/2||Gx-h||^2 s.t. Ax=b; luego
        s0 = h - G x0, z0 = -s0 y desplazamientos para estricta positividad
        (Nocedal & Wright, ec. 14.41 adaptada a QP)."""
        n, p, m = c.size, b.size, h.size
        K0 = np.zeros((n + p, n + p))
        K0[:n, :n] = Q + G.T @ G
        K0[n:, :n] = A
        K0[:n, n:] = A.T
        self._norm_K = float(np.linalg.norm(K0, np.inf))
        rhs0 = np.concatenate([-c + G.T @ h, b])
        _, sol, _ = self._factor_and_solve(K0, rhs0, self.reg_init)
        x, nu = sol[:n], sol[n:]
        if m == 0:
            return x, nu, np.zeros(0), np.zeros(0)

        s0 = h - G @ x
        z0 = -s0
        s_hat = s0 + max(-1.5 * s0.min(), 0.0)
        z_hat = z0 + max(-1.5 * z0.min(), 0.0)
        sz = float(s_hat @ z_hat)
        if sz > 0.0 and z_hat.sum() > 0.0 and s_hat.sum() > 0.0:
            s = s_hat + 0.5 * sz / z_hat.sum()
            z = z_hat + 0.5 * sz / s_hat.sum()
        else:  # degenerado: punto central genérico
            s, z = np.ones(m), np.ones(m)
        return x, nu, s, z

    # ------------------------------------------------------------------ #
    # Un paso Predictor-Corrector de Mehrotra
    # ------------------------------------------------------------------ #
    def _mehrotra_step(
        self, Q: FloatArray, A: FloatArray, G: FloatArray, K: FloatArray,
        x: FloatArray, nu: FloatArray, s: FloatArray, z: FloatArray,
        r_d: FloatArray, r_eq: FloatArray, r_in: FloatArray, mu: float,
    ) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray, Dict[str, float]]:
        n, p, m = x.size, nu.size, s.size

        # --- Ensamblaje del bloque (1,1): H = Q + G' W G, W = diag(z/s) -----
        if m > 0:
            W = z / s
            K[:n, :n] = Q + (G.T * W) @ G
        else:
            K[:n, :n] = Q
        self._norm_K = float(np.linalg.norm(K, np.inf))

        def _rhs(r_c: FloatArray) -> FloatArray:
            t = (z * r_in - r_c) / s if m > 0 else np.zeros(0)
            return np.concatenate([-r_d - G.T @ t, -r_eq])

        def _split(sol: FloatArray, r_c: FloatArray
                   ) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
            dx, dnu = sol[:n], sol[n:]
            ds = -r_in - G @ dx
            dz = (-r_c - z * ds) / s if m > 0 else np.zeros(0)
            return dx, dnu, ds, dz

        # --- Predictor (affine scaling): r_c = s∘z -------------------------
        r_c_aff = s * z if m > 0 else np.zeros(0)
        fac, sol_aff, reg_used = self._factor_and_solve(K, _rhs(r_c_aff), self._reg)
        self._reg = max(self.reg_init, reg_used / self.reg_growth)  # decaimiento
        dx, dnu, ds, dz = _split(sol_aff, r_c_aff)
        sigma, alpha_aff = 0.0, 1.0

        if m > 0:
            alpha_aff = min(1.0, _max_step(s, ds), _max_step(z, dz))
            mu_aff = float((s + alpha_aff * ds) @ (z + alpha_aff * dz)) / m
            # --- Centrado de Mehrotra: sigma = (mu_aff / mu)^3 --------------
            sigma = float(np.clip((mu_aff / mu) ** 3, 0.0, 1.0)) if mu > 0.0 else 0.0
            # --- Corrector: añade término de segundo orden y centrado -------
            r_c = s * z + ds * dz - sigma * mu
            rhs_cor = _rhs(r_c)
            sol_cor, be = self._refined_solve(K, fac, rhs_cor)
            if not (np.isfinite(be) and be <= self.resid_tol and np.all(np.isfinite(sol_cor))):
                fac, sol_cor, reg_used = self._factor_and_solve(
                    K, rhs_cor, min(reg_used * self.reg_growth, self.reg_max))
                self._reg = max(self.reg_init, reg_used / self.reg_growth)
            dx, dnu, ds, dz = _split(sol_cor, r_c)

        # --- Longitud de paso: fracción al borde + backtracking ------------
        alpha_max = min(_max_step(s, ds), _max_step(z, dz)) if m > 0 else np.inf
        alpha = min(1.0, self.step_fraction * alpha_max)
        for _ in range(self.max_backtrack):
            s_new, z_new = s + alpha * ds, z + alpha * dz
            if m == 0 or (np.all(s_new > 0.0) and np.all(z_new > 0.0)):
                break
            alpha *= 0.5
        else:
            raise QPNumericalError("Backtracking falló: no se pudo mantener s > 0 y z > 0.")

        info = {"alpha": float(alpha), "sigma": sigma, "reg": float(reg_used),
                "cond": self._last_cond}
        return x + alpha * dx, nu + alpha * dnu, s_new, z_new, info

    # ------------------------------------------------------------------ #
    # Interfaz pública
    # ------------------------------------------------------------------ #
    def fit(
        self,
        Q: FloatArray,
        c: FloatArray,
        A_eq: Optional[FloatArray],
        b_eq: Optional[FloatArray],
        A_ineq: Optional[FloatArray],
        b_ineq: Optional[FloatArray],
    ) -> Dict[str, Any]:
        """Resuelve el QP.

        Returns
        -------
        dict con:
            x_opt          : solución (n,)
            status         : 'Optimal' | 'Max_Iter_Reached' | 'Numerical_Error'
            iterations     : iteraciones de Newton ejecutadas
            kkt_residuals  : {'primal': [...], 'dual': [...], 'gap': [...]}
                             (historial; la posición k es el residuo ANTES del paso k+1)
            objective, nu_opt, z_opt, slack_opt, history (alpha, sigma, reg, cond), message

        Raises
        ------
        QPInputError : datos inconsistentes o Q no convexa.
        """
        Q_orig, c_orig, A, b, G, h = self._validate(Q, c, A_eq, b_eq, A_ineq, b_ineq)
        kappa = 1.0
        if self.scale_cost:
            nrm = max(float(np.linalg.norm(Q_orig, np.inf)), _inf_norm(c_orig))
            if nrm > 0.0:
                kappa = 1.0 / nrm
        Q, c = kappa * Q_orig, kappa * c_orig
        n, p, m = c.size, b.size, h.size
        self._reg = self.reg_init
        self._reg_signs = np.concatenate([np.ones(n), -np.ones(p)])
        self._last_cond = float("nan")

        hist: Dict[str, List[float]] = {"primal": [], "dual": [], "gap": []}
        steps: Dict[str, List[float]] = {"alpha": [], "sigma": [], "reg": [], "cond": []}
        status, message = "Max_Iter_Reached", "Se alcanzó max_iter sin converger."

        # Escalas para tolerancias relativas (opcional)
        sc_p = 1.0 + max(_inf_norm(b), _inf_norm(h)) if self.relative_tol else 1.0
        sc_d = 1.0 + _inf_norm(c) if self.relative_tol else 1.0

        try:
            x, nu, s, z = self._initial_point(Q, c, A, b, G, h)
        except QPNumericalError as exc:
            return self._result(np.full(n, np.nan), np.full(p, np.nan), np.zeros(m),
                                np.zeros(m), Q_orig, c_orig, "Numerical_Error", 0, hist, steps,
                                f"Fallo en el punto inicial: {exc}")

        K = np.zeros((n + p, n + p))
        K[n:, :n] = A
        K[:n, n:] = A.T

        if self.verbose:
            print(f"{'it':>3} {'primal':>10} {'dual':>10} {'gap':>10} {'alpha':>8} {'sigma':>8} {'reg':>8}")

        k = 0
        for k in range(self.max_iter + 1):
            r_d = Q @ x + c + A.T @ nu + G.T @ z
            r_eq = A @ x - b
            r_in = G @ x + s - h
            pres = max(_inf_norm(r_eq), _inf_norm(r_in))
            dres = _inf_norm(r_d)
            mu = float(s @ z) / m if m > 0 else 0.0
            hist["primal"].append(pres)
            hist["dual"].append(dres)
            hist["gap"].append(mu)

            if self.verbose:
                a_, sg_, rg_ = (steps["alpha"][-1], steps["sigma"][-1], steps["reg"][-1]) if steps["alpha"] else (np.nan,) * 3
                print(f"{k:3d} {pres:10.3e} {dres:10.3e} {mu:10.3e} {a_:8.2e} {sg_:8.2e} {rg_:8.1e}")

            if not (np.isfinite(pres) and np.isfinite(dres) and np.isfinite(mu)):
                status, message = "Numerical_Error", "Residuos no finitos."
                break
            if max(_inf_norm(x), _inf_norm(nu), _inf_norm(z)) > self.divergence_limit:
                status, message = ("Numerical_Error",
                                   "Divergencia de iterados: posible problema infactible o no acotado.")
                break

            obj = 0.5 * x @ Q @ x + c @ x
            tol_g = self.tol * (1.0 + abs(obj)) if self.relative_tol else self.tol
            if pres < self.tol * sc_p and dres < self.tol * sc_d and mu < tol_g:
                status, message = "Optimal", "Convergencia alcanzada."
                break
            if k == self.max_iter:
                break

            try:
                x, nu, s, z, info = self._mehrotra_step(Q, A, G, K, x, nu, s, z,
                                                        r_d, r_eq, r_in, mu)
            except QPNumericalError as exc:
                status, message = "Numerical_Error", str(exc)
                break
            for key in steps:
                steps[key].append(info[key])
            if info["alpha"] < 1e-12:
                status, message = "Numerical_Error", "Paso de longitud ~0 (estancamiento)."
                break

        return self._result(x, nu / kappa, z / kappa, s, Q_orig, c_orig,
                            status, k, hist, steps, message)

    @staticmethod
    def _result(x: FloatArray, nu: FloatArray, z: FloatArray, s: FloatArray,
                Q: FloatArray, c: FloatArray, status: str, iters: int,
                hist: Dict[str, List[float]], steps: Dict[str, List[float]],
                message: str) -> Dict[str, Any]:
        obj = float(0.5 * x @ Q @ x + c @ x) if np.all(np.isfinite(x)) else float("nan")
        return {
            "x_opt": x,
            "status": status,
            "iterations": iters,
            "kkt_residuals": hist,
            "objective": obj,
            "nu_opt": nu,
            "z_opt": z,
            "slack_opt": s,
            "history": steps,
            "message": message,
        }


# --------------------------------------------------------------------------- #
# Integración: QP de portafolio de mínima varianza con presupuesto de turnover
# --------------------------------------------------------------------------- #
def build_portfolio_qp(
    Sigma: FloatArray,
    w_prev_drift: FloatArray,
    L: float,
    R_target: Optional[float] = None,
    mu: Optional[FloatArray] = None,
    *,
    long_only: bool = True,
    w_max: Optional[float | FloatArray] = None,
    gamma: float = 0.0,
) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Construye (Q, c, A_eq, b_eq, A_ineq, b_ineq) para `PrimalDualQPSolver`.

    Variables  x = [w ; s] ∈ R^{2N},  w = pesos,  s = cota del |cambio| por activo:

        min   w' Σ w  - gamma * mu' w            (Q_ww = 2Σ  ⇒  obj = varianza)
        s.t.  1' w = 1                           (presupuesto)
              w - s <=  w_prev                   ┐ s_i >= |w_i - w_prev_i|
             -w - s <= -w_prev                   ┘
              1' s  <= L                         (turnover L1:  ||w - w_prev||_1 <= L)
             -w <= 0                             (si long_only)
              w <= w_max                         (si w_max no es None)
             -mu' w <= -R_target                 (si R_target y mu dados: retorno duro)

    Parameters
    ----------
    Sigma : (N,N) covarianza estimada Σ_t (simétrica PSD).
    w_prev_drift : (N,) pesos previos ajustados por deriva w_{t-1}^+.
    L : presupuesto de turnover, L > 0 (L = 0 no tiene interior estricto).
    R_target, mu : retorno objetivo y retornos esperados (opcionales). Con ambos
        se impone la restricción dura  mu'w >= R_target.
    gamma : peso de la versión "suave" del retorno (término lineal -gamma*mu'w).
        Requiere `mu`. Con gamma=0 es mínima varianza pura.
    long_only, w_max : cotas opcionales sobre w.

    Raises
    ------
    QPInputError : dimensiones inconsistentes, L <= 0, etc.
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    w_prev = np.asarray(w_prev_drift, dtype=np.float64).ravel()
    N = w_prev.size
    if Sigma.shape != (N, N):
        raise QPInputError(f"Sigma debe ser ({N},{N}); recibido {Sigma.shape}.")
    if not (np.isfinite(L) and L > 0.0):
        raise QPInputError("El presupuesto de turnover L debe ser finito y > 0.")
    if (R_target is not None or gamma != 0.0) and mu is None:
        raise QPInputError("Se requiere `mu` para R_target o gamma != 0.")
    if mu is not None:
        mu = np.asarray(mu, dtype=np.float64).ravel()
        if mu.size != N:
            raise QPInputError(f"mu debe tener {N} elementos.")

    I, Z = np.eye(N), np.zeros((N, N))
    ones, zeros = np.ones((1, N)), np.zeros((1, N))

    # Objetivo
    Q = np.block([[2.0 * Sigma, Z], [Z, Z]])
    c = np.concatenate([-gamma * mu if (mu is not None and gamma != 0.0) else np.zeros(N),
                        np.zeros(N)])

    # Igualdad: suma de pesos = 1
    A_eq = np.hstack([ones, zeros])
    b_eq = np.array([1.0])

    # Desigualdades
    rows: List[FloatArray] = [
        np.hstack([I, -I]),          #  w - s <=  w_prev
        np.hstack([-I, -I]),         # -w - s <= -w_prev
        np.hstack([zeros, ones]),    #  1's   <= L
    ]
    rhs: List[FloatArray] = [w_prev, -w_prev, np.array([L])]
    if long_only:
        rows.append(np.hstack([-I, Z]))
        rhs.append(np.zeros(N))
    if w_max is not None:
        rows.append(np.hstack([I, Z]))
        rhs.append(np.broadcast_to(np.asarray(w_max, dtype=np.float64), (N,)).copy())
    if R_target is not None:
        rows.append(np.hstack([-mu[None, :], zeros]))
        rhs.append(np.array([-float(R_target)]))

    return Q, c, A_eq, b_eq, np.vstack(rows), np.concatenate(rhs)


def split_solution(x_opt: FloatArray, N: int) -> Tuple[FloatArray, FloatArray]:
    """Separa x = [w ; s] en (pesos w, cotas de turnover s)."""
    return x_opt[:N], x_opt[N:]


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, T = 20, 250
    F = rng.standard_normal((T, N)) * 0.01
    Sigma_hat = np.cov(F, rowvar=False) + 1e-6 * np.eye(N)
    w_prev = rng.dirichlet(np.ones(N))      # w_{t-1}^+ (suma 1)
    mu_hat = rng.standard_normal(N) * 1e-3
    L_budget = 0.30

    Q, c, A_eq, b_eq, A_in, b_in = build_portfolio_qp(
        Sigma_hat, w_prev, L_budget, R_target=None, mu=mu_hat, gamma=0.0)

    solver = PrimalDualQPSolver(max_iter=100, verbose=True)
    out = solver.fit(Q, c, A_eq, b_eq, A_in, b_in)
    w, s = split_solution(out["x_opt"], N)

    print("\nstatus      :", out["status"], f"({out['iterations']} iteraciones)")
    print("varianza    : %.6e" % (w @ Sigma_hat @ w))
    print("sum(w)      : %.12f" % w.sum())
    print("min(w)      : %.3e" % w.min())
    print("turnover L1 : %.6f  (L = %.2f)" % (np.abs(w - w_prev).sum(), L_budget))