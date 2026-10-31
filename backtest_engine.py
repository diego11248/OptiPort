"""
backtest_engine.py
==================
Motor único de simulación, usado por `backtest.py` y `sensitivity.py`.

Orden temporal correcto en cada día t (corrige el sesgo de look-ahead):

    1. Se GANA el retorno r_t con los pesos w que ya se tenían al abrir el día.
    2. Los pesos DERIVAN con r_t:  w_drift = w ∘ (1 + r_t) / (1 + w'r_t).
    3. Al cierre se REBALANCEA con Σ = cov(t+1) (ventana que termina en t, ya
       conocida) anclado a w_drift. El resultado w_new se mantiene el día t+1.

Covarianzas: ya NO se carga un array (T, N, N) (≈1 GB con N=500). `MarketData.cov(t)`
lee Σ_t de un caché float32 con memmap si existe (config.COV_CACHE_FILE) o la
recalcula con Ledoit-Wolf sobre la ventana [t, t+W) de los retornos.

Convención de turnover:  turnover_t = 0.5 * ||w_new - w_drift||_1  (unidireccional).
El presupuesto del solver es  ||Δw||_1 <= 2 * turnover_limit.
El costo se cobra sobre el nocional negociado:  cost_rate * ||Δw||_1.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.covariance import ledoit_wolf

import config
from pdipm_portfolio import PortfolioQPSolver
from pdipm_qp_solver import (PrimalDualQPSolver, QPNumericalError,
                             build_portfolio_qp)

FloatArray = NDArray[np.float64]
InitMode = Literal["minvar", "equal"]


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2)
def _open_cov_cache(path: str) -> np.ndarray:
    """Abre el caché de covarianzas como memmap de solo lectura (no copia el archivo)."""
    return np.load(path, mmap_mode="r")


@dataclass(frozen=True)
class MarketData:
    dates: pd.DatetimeIndex
    tickers: List[str]
    window: int
    all_returns: FloatArray          # (window + T, N): historia completa para estimar Σ
    returns: FloatArray              # (T, N); returns[t] = retorno del día dates[t]
    benchmark: Optional[pd.Series]   # retornos de SPY alineados a `dates`
    cov_cache_path: Optional[str] = None   # solo la ruta: el dataclass sigue siendo liviano y serializable

    def cov(self, t: int) -> FloatArray:
        """Σ para dates[t]: Ledoit-Wolf sobre los retornos de [dates[t]-W, dates[t]) (float64)."""
        T = self.returns.shape[0]
        if not (0 <= t < T):
            raise IndexError(f"t={t} fuera de rango [0, {T}).")
        if self.cov_cache_path is not None:
            return np.array(_open_cov_cache(self.cov_cache_path)[t], dtype=np.float64)
        return ledoit_wolf(self.all_returns[t:t + self.window])[0]


def load_market_data(rolling_file: Path = config.ROLLING_DATA_FILE,
                     returns_file: Path = config.RETURNS_FILE,
                     benchmark_file: Path = config.BENCHMARK_FILE,
                     cov_cache_file: Optional[Path] = config.COV_CACHE_FILE) -> MarketData:
    """Carga y valida los datos; alinea retornos y fechas una sola vez."""
    with np.load(rolling_file) as f:
        dates = pd.to_datetime(f["dates"])
        tickers = [str(t) for t in f["tickers"]]
        window = int(f["window"])

    ret_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    missing_cols = [t for t in tickers if t not in ret_df.columns]
    if missing_cols:
        raise ValueError(f"{returns_file} no contiene {len(missing_cols)} activos de {rolling_file}.")
    ret_df = ret_df[tickers]
    if len(ret_df) != window + len(dates) or not ret_df.index[window:].equals(dates):
        raise ValueError(f"{returns_file} y {rolling_file} no corresponden a la misma corrida "
                         "de dataset.py (fechas distintas); vuelve a generarlos.")
    all_R = ret_df.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(all_R)):
        raise ValueError("Los retornos contienen NaN/Inf.")
    T, N = len(dates), len(tickers)
    if T < 2:
        raise ValueError("Se necesitan al menos 2 fechas.")

    bench: Optional[pd.Series] = None
    if Path(benchmark_file).exists():
        bench = pd.read_csv(benchmark_file, index_col=0, parse_dates=True).iloc[:, 0].reindex(dates)

    cache: Optional[str] = None
    if cov_cache_file is not None and Path(cov_cache_file).exists():
        mm = _open_cov_cache(str(cov_cache_file))
        probe = ledoit_wolf(all_R[0:window])[0]
        stale = (mm.shape != (T, N, N)
                 or not np.allclose(mm[0], probe, rtol=1e-4, atol=1e-9 * max(1.0, np.abs(probe).max())))
        if stale:
            print(f"Aviso: {cov_cache_file} no corresponde a estos datos; se ignora (Σ se calcula al vuelo).")
        else:
            cache = str(cov_cache_file)

    return MarketData(dates, tickers, window, all_R, all_R[window:], bench, cache)


# --------------------------------------------------------------------------- #
# Simulación
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BacktestResult:
    dates: pd.DatetimeIndex
    tickers: List[str]
    weights: FloatArray              # (T, N): pesos mantenidos DURANTE el día t
    gross_returns: FloatArray        # (T,)
    net_returns: FloatArray          # (T,)
    turnover: FloatArray             # (T,): 0.5*||w_new - w_drift||_1 al cierre de t
    iterations: NDArray[np.intp]     # (T,): 0 si ese día no se resolvió un QP
    failed: NDArray[np.bool_]        # (T,): True si se mantuvo w_drift por fallo


class PortfolioSolver(Protocol):
    """Interfaz mínima que usa el motor: resuelve el QP de un rebalanceo."""
    def solve(self, Sigma: FloatArray, w_prev: FloatArray, L: float) -> Dict[str, Any]: ...


class GenericSolverAdapter:
    """Adapta `PrimalDualQPSolver` (KKT densa genérica) a la interfaz `solve`."""

    def __init__(self, solver: PrimalDualQPSolver) -> None:
        self._solver = solver

    def solve(self, Sigma: FloatArray, w_prev: FloatArray, L: float) -> Dict[str, Any]:
        return self._solver.fit(*build_portfolio_qp(Sigma, w_prev, L, long_only=True))


def make_solver(kind: str = config.SOLVER) -> PortfolioSolver:
    """'structured' (por defecto, N≈500) o 'generic' (referencia; ~10x más lento en N=500)."""
    if kind == "structured":
        return PortfolioQPSolver(tol=1e-8, max_iter=100, scale_cost=True, verbose=False)
    if kind == "generic":
        # check_psd=False: Q = blockdiag(2Σ, 0) con Σ de Ledoit-Wolf es PSD por construcción.
        return GenericSolverAdapter(PrimalDualQPSolver(tol=1e-8, max_iter=100, scale_cost=True,
                                                       check_psd=False, verbose=False))
    raise ValueError("kind debe ser 'structured' o 'generic'.")


def _rebalance(Sigma: FloatArray, w_prev: FloatArray, l1_limit: float,
               solver: PortfolioSolver) -> Tuple[FloatArray, bool, int]:
    """Resuelve el QP; ante un fallo NUMÉRICO devuelve (w_prev, False, iters).

    Los errores de entrada (QPInputError) NO se capturan: son bugs, no fallos del solver.
    """
    N = w_prev.size
    try:
        res = solver.solve(Sigma, w_prev, l1_limit)
    except QPNumericalError:
        return w_prev, False, 0
    iters = int(res["iterations"])
    if res["status"] != "Optimal":
        return w_prev, False, iters
    w = np.clip(res["x_opt"][:N], 0.0, None)   # elimina negativos ~ -1e-9 del solver
    total = w.sum()
    if not np.isfinite(total) or total <= 0.0:
        return w_prev, False, iters
    return w / total, True, iters


def run_backtest(data: MarketData, turnover_limit: float,
                 cost_rate: float = 0.0, init: InitMode = "minvar",
                 solver: Optional[PortfolioSolver] = None,
                 max_fail_frac: float = config.MAX_FAIL_FRAC) -> BacktestResult:
    """Simula la estrategia de mínima varianza con presupuesto de turnover.

    Parameters
    ----------
    turnover_limit : cota de 0.5*||Δw||_1 por rebalanceo, en (0, 1].
    cost_rate      : costo por unidad de nocional negociado (por lado).
    max_fail_frac  : fracción máxima tolerada de rebalanceos fallidos (si no, RuntimeError).
    init           : 'minvar' arranca en el portafolio de mínima varianza SIN
                     restricción de turnover (Σ = cov(0)); así todos los L
                     comparten el mismo punto de partida y no se mide la
                     transición desde 1/N. 'equal' arranca en 1/N.
    """
    if not (0.0 < turnover_limit <= 1.0):
        raise ValueError("turnover_limit debe estar en (0, 1].")
    if cost_rate < 0.0:
        raise ValueError("cost_rate no puede ser negativo.")

    solver = solver if solver is not None else make_solver()
    T, N = data.returns.shape
    l1_limit = 2.0 * turnover_limit

    w = np.full(N, 1.0 / N)
    if init == "minvar":
        w, ok, _ = _rebalance(data.cov(0), w, 2.0, solver)
        if not ok:
            w = np.full(N, 1.0 / N)
    elif init != "equal":
        raise ValueError("init debe ser 'minvar' o 'equal'.")

    weights = np.empty((T, N))
    gross = np.empty(T)
    net = np.empty(T)
    turnover = np.zeros(T)
    iterations = np.zeros(T, dtype=np.intp)
    failed = np.zeros(T, dtype=bool)

    for t in range(T):
        r = data.returns[t]
        weights[t] = w

        # 1) ganar r_t con los pesos que ya se tenían
        growth = 1.0 + float(w @ r)
        gross[t] = growth - 1.0

        # 2) deriva
        w_drift = w * (1.0 + r) / growth

        # 3) rebalanceo al cierre con Σ que ya incluye r_t (cov(t+1))
        if t < T - 1:
            w_new, ok, iters = _rebalance(data.cov(t + 1), w_drift, l1_limit, solver)
            iterations[t] = iters
            failed[t] = not ok
        else:
            w_new = w_drift

        l1 = float(np.abs(w_new - w_drift).sum())
        turnover[t] = 0.5 * l1
        net[t] = growth * (1.0 - cost_rate * l1) - 1.0
        w = w_new

    fail_frac = failed.sum() / max(T - 1, 1)
    if fail_frac > max_fail_frac:
        raise RuntimeError(f"El solver falló en {fail_frac:.1%} de los rebalanceos "
                           f"(límite {max_fail_frac:.1%}); resultados no confiables.")
    return BacktestResult(data.dates, data.tickers, weights, gross, net,
                          turnover, iterations, failed)


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def compute_metrics(returns: ArrayLike,
                    rf_annual: float = config.RISK_FREE_ANNUAL,
                    periods: int = config.TRADING_DAYS) -> Dict[str, float]:
    """Métricas anualizadas de una serie de retornos diarios."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 2:
        raise ValueError("Se necesitan al menos 2 retornos.")
    years = n / periods

    ann_ret = float(r.mean() * periods)
    vol = float(r.std(ddof=1) * np.sqrt(periods))
    sharpe = (ann_ret - rf_annual) / vol if vol > 0.0 else float("nan")

    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return {
        "ann_return": ann_ret,
        "ann_return_geo": float(wealth[-1] ** (1.0 / years) - 1.0),
        "ann_vol": vol,
        "sharpe": sharpe,
        # Error estándar aprox. del Sharpe anual con retornos i.i.d. (Lo, 2002)
        "sharpe_se": float(np.sqrt((1.0 + 0.5 * sharpe ** 2) / years)),
        "max_drawdown": float((wealth / peak - 1.0).min()),
    }


def reference_returns(data: MarketData) -> Dict[str, FloatArray]:
    """Series de referencia: 1/N rebalanceado a diario y SPY (si existe)."""
    refs: Dict[str, FloatArray] = {"1/N diario": data.returns.mean(axis=1)}
    if data.benchmark is not None and data.benchmark.notna().all():
        refs["SPY"] = data.benchmark.to_numpy(dtype=np.float64)
    return refs