"""
benchmark.py
============
Compara a lo largo del histórico de rebalanceos en horizonte rodante:
  1. PDIPM propio (config.SOLVER: 'structured' por defecto)
  2. SLSQP (SciPy)         -> se omite si N > config.SLSQP_MAX_N (inviable con N≈500)
  3. CVXPY (Clarabel)

    python benchmark.py                      # todos los días
    python benchmark.py --max-days 60        # muestra de los primeros 60 rebalanceos
    python benchmark.py --solver generic     # PDIPM genérico (referencia lenta)
    python benchmark.py --with-slsqp         # fuerza SLSQP aunque N sea grande (minutos por día)

Escribe benchmark_3way_summary.csv (tabla legible), benchmark_3way_numeric.csv
(cifras para el dashboard) y benchmark_3way_comparison.png.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config
from backtest_engine import load_market_data, make_solver

FloatArray = np.ndarray
# runner(Sigma, w_drift, L, l1_limit) -> (w_new, total_ms, pure_ms, ok, iters)
Runner = Callable[[FloatArray, FloatArray, float, float], Tuple[FloatArray, float, float, bool, int]]


def _normalize(w: FloatArray) -> Optional[FloatArray]:
    w = np.clip(w, 0.0, None)
    total = w.sum()
    return w / total if np.isfinite(total) and total > 0 else None


def make_pdipm_runner(kind: str) -> Runner:
    solver = make_solver(kind)

    def run(Sigma: FloatArray, w_drift: FloatArray, L: float, l1: float):
        t0 = time.perf_counter()
        try:
            res = solver.solve(Sigma, w_drift, l1)
        except Exception:
            return w_drift, (time.perf_counter() - t0) * 1e3, 0.0, False, 0
        ms = (time.perf_counter() - t0) * 1e3
        w = _normalize(res["x_opt"][:w_drift.size]) if res["status"] == "Optimal" else None
        if w is None:
            return w_drift, ms, ms, False, int(res["iterations"])
        return w, ms, ms, True, int(res["iterations"])
    return run


def run_cvxpy(Sigma: FloatArray, w_drift: FloatArray, L: float, l1: float, solver=cp.CLARABEL):
    """Total = compilación del DSL + solución; 'pure' = tiempo reportado por el solver."""
    N = w_drift.size
    w = cp.Variable(N)
    # Σ en unidades de %² (×1e4): con Σ ~ 1e-4 el objetivo es ~1e-7 y Clarabel (tolerancias
    # absolutas) se detiene ~0.02-0.04 % por encima del óptimo; escalada llega a ~1e-7 relativo.
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma * 1e4))),
                      [cp.sum(w) == 1.0, w >= 0.0, 0.5 * cp.norm(w - w_drift, 1) <= L])
    t0 = time.perf_counter()
    try:
        prob.solve(solver=solver, verbose=False)
        total = (time.perf_counter() - t0) * 1e3
        pure = prob.solver_stats.solve_time * 1e3 if prob.solver_stats.solve_time is not None else total
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w.value is not None:
            w_new = _normalize(np.asarray(w.value))
            if w_new is not None:
                return w_new, total, pure, True, 0
    except Exception:
        pass
    return w_drift, 0.0, 0.0, False, 0


def run_slsqp(Sigma: FloatArray, w_drift: FloatArray, L: float, l1: float):
    N = w_drift.size
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: L - 0.5 * np.sum(np.abs(w - w_drift))}]
    t0 = time.perf_counter()
    res = minimize(lambda w: float(w @ Sigma @ w), w_drift, method="SLSQP",
                   jac=lambda w: 2.0 * (Sigma @ w), bounds=[(0.0, None)] * N,
                   constraints=cons, options={"ftol": 1e-9, "maxiter": 500})
    ms = (time.perf_counter() - t0) * 1e3
    w_new = _normalize(res.x) if res.success else None
    return (w_new, ms, ms, True, 0) if w_new is not None else (w_drift, ms, ms, False, 0)


@dataclass
class Track:
    name: str
    runner: Runner
    w: FloatArray
    total_ms: List[float] = field(default_factory=list)
    pure_ms: List[float] = field(default_factory=list)
    obj: List[float] = field(default_factory=list)
    iters: List[int] = field(default_factory=list)
    fails: int = 0


def run_benchmark(max_days: Optional[int], solver_kind: str, with_slsqp: bool) -> None:
    data = load_market_data()
    T, N = data.returns.shape
    n_days = T - 1 if max_days is None else min(max_days, T - 1)
    L = config.TURNOVER_BUDGET
    l1 = 2.0 * L

    w0 = np.full(N, 1.0 / N)
    tracks: Dict[str, Track] = {"pdipm": Track(f"PDIPM propio ({solver_kind})", make_pdipm_runner(solver_kind), w0.copy()),
                                "cvxpy": Track("CVXPY (Clarabel)", run_cvxpy, w0.copy())}
    use_slsqp = with_slsqp or N <= config.SLSQP_MAX_N
    if use_slsqp:
        tracks["slsqp"] = Track("SLSQP (SciPy)", run_slsqp, w0.copy())
    else:
        print(f"Aviso: SLSQP omitido (N={N} > {config.SLSQP_MAX_N}); usa --with-slsqp para forzarlo.")

    print(f"--- Benchmark: {n_days} días | N={N} activos | L={L} | solver propio: {solver_kind} ---")
    for t in range(n_days):
        Sigma = data.cov(t + 1)
        r = data.returns[t]
        for tr in tracks.values():
            drift = tr.w * (1.0 + r) / (1.0 + float(tr.w @ r))     # cada simulador deriva por su cuenta
            w_new, tot, pure, ok, it = tr.runner(Sigma, drift, L, l1)
            tr.fails += 0 if ok else 1
            tr.w = w_new if ok else drift
            tr.total_ms.append(tot); tr.pure_ms.append(pure); tr.iters.append(it)
            tr.obj.append(float(tr.w @ Sigma @ tr.w))
        if (t + 1) % 50 == 0:
            print(f"  {t + 1}/{n_days} días")

    # ---------------------------- Resumen ---------------------------------
    base = np.array(tracks["cvxpy"].obj)

    def gap(tr: Track) -> np.ndarray:
        return np.abs(np.array(tr.obj) - base) / np.maximum(base, 1e-12)

    g_p = gap(tracks["pdipm"])
    rows = {"Métrica": ["Tiempo promedio solución (ms)", "Tiempo máximo solución (ms)",
                        "Overhead de compilación (ms)", "Fallos / infactibilidad",
                        "Gap relativo de varianza vs CVXPY (promedio)",
                        "Gap relativo de varianza vs CVXPY (máximo)", "Iteraciones promedio"]}
    numeric = []
    for key, tr in tracks.items():
        tot, pure = np.array(tr.total_ms), np.array(tr.pure_ms)
        g = gap(tr)
        overhead = float(tot.mean() - pure.mean()) if key == "cvxpy" else 0.0
        iters = [i for i in tr.iters if i > 0]
        rows[tr.name] = [
            f"{tot.mean():.2f} ms",
            f"{tot.max():.2f} ms",
            f"{overhead:.2f} ms" if key == "cvxpy" else ("0.00 ms (sin overhead)" if key == "pdipm" else "N/A"),
            f"{tr.fails} / {n_days}",
            "0.00 (base)" if key == "cvxpy" else f"{g.mean():.2e}",
            "0.00 (base)" if key == "cvxpy" else f"{g.max():.2e}",
            f"{np.mean(iters):.1f}" if iters else "N/A",
        ]
        numeric.append({"Solver": tr.name, "Tiempo Total (ms)": float(tot.mean()),
                        "Tiempo Solver Puro (ms)": float(pure.mean()),
                        "Tasa de Convergencia (%)": 100.0 * (1 - tr.fails / n_days),
                        "Fallos": tr.fails, "Días": n_days, "N": N})
    summary = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("COMPARATIVA DE DESEMPEÑO")
    print("=" * 100)
    print(summary.to_string(index=False))
    summary.to_csv("benchmark_3way_summary.csv", index=False)
    pd.DataFrame(numeric).to_csv("benchmark_3way_numeric.csv", index=False)

    # ---------------------------- Gráficas --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    labels = [tr.name for tr in tracks.values()]
    axes[0].boxplot([tr.total_ms for tr in tracks.values()])
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Tiempo por rebalanceo (ms)")
    axes[0].set_title(f"Distribución de tiempos (N={N})")
    axes[0].grid(True)
    axes[1].plot(data.dates[1:n_days + 1], np.maximum(g_p, 1e-18), color="darkgreen", lw=1.2,
                 label="|f_PDIPM − f_CVXPY| / f_CVXPY")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Gap relativo")
    axes[1].set_title("Precisión relativa respecto a CVXPY")
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1].legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("benchmark_3way_comparison.png", dpi=150)
    print("\nGráfica guardada en 'benchmark_3way_comparison.png'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--solver", choices=["structured", "generic"], default=config.SOLVER)
    ap.add_argument("--with-slsqp", action="store_true")
    args = ap.parse_args()
    run_benchmark(args.max_days, args.solver, args.with_slsqp)
