"""Pruebas del solver estructurado y del pipeline N≈500 (sin red: datos sintéticos).
    python test_structured.py            # rápido (N<=60 y un mini-universo sintético de N=120)
    python test_structured.py --big      # añade un backtest sintético de N=500
"""
import gc
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

import dataset
from backtest_engine import (_open_cov_cache, compute_metrics, load_market_data, make_solver,
                             run_backtest)
from pdipm_portfolio import PortfolioQPSolver
from pdipm_qp_solver import PrimalDualQPSolver, build_portfolio_qp


def random_cov(rng, N, T=252, k=3):
    f = rng.standard_normal((T, k)) @ rng.standard_normal((k, N)) * 0.006
    R = f + rng.standard_normal((T, N)) * rng.uniform(0.008, 0.025, N)
    return ledoit_wolf(R)[0]


def test_vs_generic():
    rng = np.random.default_rng(0)
    gen, st = PrimalDualQPSolver(tol=1e-8, check_psd=False), PortfolioQPSolver(tol=1e-8)
    worst = 0.0
    for N in (8, 30, 60):
        S = random_cov(rng, N)
        for zeros in (False, True):
            w0 = rng.dirichlet(np.ones(N)) if not zeros else np.where(rng.random(N) < 0.7, 0, rng.random(N))
            w0 = w0 / w0.sum()
            for L in (0.01, 0.2, 2.0):
                a = gen.fit(*build_portfolio_qp(S, w0, L)); b = st.solve(S, w0, L)
                assert a["status"] == b["status"] == "Optimal", (N, zeros, L, a["message"], b["message"])
                wa, wb = a["x_opt"][:N], b["x_opt"][:N]
                assert abs(wb.sum() - 1) < 1e-9 and wb.min() > -1e-9
                assert np.abs(wb - w0).sum() <= L + 1e-7
                fa, fb = wa @ S @ wa, wb @ S @ wb
                # El estructurado (escalado 1/N) nunca debe quedar peor que el genérico, y ambos coinciden.
                assert fb <= fa * (1 + 1e-9), f"estructurado peor que genérico (N={N}, L={L})"
                rel = abs(fb - fa) / fa
                assert rel < 1e-3, f"objetivo difiere {rel:.2e} (N={N}, L={L})"  # el genérico para antes (tolerancia absoluta)
                worst = max(worst, rel)
    print(f"OK  estructurado ≈ genérico y nunca peor (mayor diferencia relativa del objetivo: {worst:.1e})")


def test_optimality_large():
    """Regresión: con N grande el solver debe llegar al óptimo (antes paraba 0.4 %/10 %/45 % arriba)."""
    try:
        import cvxpy as cp
    except ImportError:
        print("--  optimalidad vs CVXPY omitida (cvxpy no instalado)")
        return
    rng = np.random.default_rng(11)
    st = PortfolioQPSolver(tol=1e-8)
    for N in (100, 500):
        S = random_cov(rng, N)
        w0 = rng.dirichlet(np.ones(N) * 3)
        for L in (0.01, 0.05, 0.5):
            r = st.solve(S, w0, 2 * L)
            assert r["status"] == "Optimal", r["message"]
            x = np.clip(r["x_opt"][:N], 0, None); x /= x.sum()
            w = cp.Variable(N)
            cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(S * 1e4))),
                       [cp.sum(w) == 1, w >= 0, 0.5 * cp.norm(w - w0, 1) <= L]).solve(
                solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
            ref = w.value @ S @ w.value
            gap = (x @ S @ x - ref) / ref
            assert gap < 1e-5, f"subóptimo por {gap:.1e} (N={N}, L={L})"
            assert 0.5 * np.abs(x - w0).sum() <= L + 1e-7
    print("OK  óptimo coincide con CVXPY de alta precisión (N=100 y N=500, gap < 1e-5)")


def make_synthetic(tmp: Path, N, T_total, seed=0, window=252):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((T_total, 3)) @ rng.standard_normal((3, N)) * 0.006
    R = f + rng.standard_normal((T_total, N)) * rng.uniform(0.008, 0.025, N)   # media 0: no hay alfa
    idx = pd.bdate_range("2023-01-02", periods=T_total)
    tickers = [f"S{i:03d}" for i in range(N)]
    pd.DataFrame(R, index=idx, columns=tickers).to_csv(tmp / "returns.csv")
    pd.Series(R.mean(axis=1), index=idx, name="SPY").to_csv(tmp / "bench.csv")
    cache = tmp / "covs.npy"
    intens = dataset.build_cov_cache(R, window, str(cache))
    dates = np.array([str(d.date()) for d in idx[window:]])
    np.savez_compressed(tmp / "meta.npz", dates=dates, tickers=np.array(tickers),
                        window=np.array(window), intensities=intens)
    return tmp / "meta.npz", tmp / "returns.csv", tmp / "bench.csv", cache


def test_pipeline(N, T_total):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        meta, ret, bench, cache = make_synthetic(tmp, N, T_total)
        data_c = load_market_data(meta, ret, bench, cache)
        data_f = load_market_data(meta, ret, bench, tmp / "no_existe.npy")
        assert data_c.cov_cache_path is not None and data_f.cov_cache_path is None
        t = 5
        a, b = data_c.cov(t), data_f.cov(t)
        assert np.allclose(a, b, rtol=1e-5, atol=1e-10), "caché float32 != Ledoit-Wolf al vuelo"
        # convención temporal: cov(t) usa SOLO datos anteriores a dates[t]
        assert np.allclose(b, ledoit_wolf(data_f.all_returns[t:t + 252])[0])
        print(f"OK  caché de Σ coherente (N={N}); memmap float32: {cache.stat().st_size / 1e6:.0f} MB")

        solver = make_solver()
        res = {}
        for L in (0.02, 1.0):
            r = run_backtest(data_c, L, cost_rate=5e-4, solver=solver)
            assert not r.failed.any()
            assert r.turnover.max() <= L + 1e-7, "turnover excede el presupuesto"
            assert np.allclose(r.weights.sum(1), 1.0, atol=1e-9) and r.weights.min() >= 0
            res[L] = r
        # Sin alfa (media 0) y con el orden temporal correcto, L pequeño NO debe inflar el retorno bruto.
        # (El orden viejo daba ~ +4.9 % anual de sesgo mecánico con L pequeño.)
        g = {L: res[L].gross_returns.mean() * 252 for L in res}
        se = res[1.0].gross_returns.std(ddof=1) * np.sqrt(252) / np.sqrt(len(res[1.0].gross_returns) / 252)
        assert abs(g[0.02] - g[1.0]) < 3 * se, f"posible look-ahead: {g}"
        print(f"OK  backtest N={N}: retorno bruto anual L=0.02 → {g[0.02]:+.3f}, L=1.0 → {g[1.0]:+.3f} "
              f"(SE≈{se:.3f}); iteraciones medias {res[0.02].iterations[res[0.02].iterations > 0].mean():.1f}")
        # En Windows no se puede borrar un archivo abierto: cerrar el memmap del caché antes de
        # que TemporaryDirectory elimine la carpeta.
        _open_cov_cache.cache_clear()
        gc.collect()


if __name__ == "__main__":
    test_vs_generic()
    test_optimality_large()
    test_pipeline(N=120, T_total=252 + 60)
    if "--big" in sys.argv:
        test_pipeline(N=500, T_total=252 + 40)
    print("Todas las pruebas pasaron.")
