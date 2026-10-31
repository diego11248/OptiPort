"""Configuración central del proyecto."""
from pathlib import Path

# --- Universo ---------------------------------------------------------------
# "sp500" (≈500 activos, constituyentes actuales) o "sp100" (tu lista de 100).
# Los nombres de archivo se derivan de aquí: ya no hay que mantener N a mano.
UNIVERSE = "sp500"
UNIVERSE_LABEL = {"sp500": "S&P 500", "sp100": "S&P 100 (selección propia)"}.get(UNIVERSE, UNIVERSE)

ROLLING_DATA_FILE = Path(f"rolling_data_{UNIVERSE}.npz")   # metadatos livianos (fechas, tickers, ventana)
RETURNS_FILE = Path(f"returns_{UNIVERSE}.csv")
UNIVERSE_FILE = Path(f"universe_{UNIVERSE}.csv")           # Symbol, Security, GICS Sector
COV_CACHE_FILE = Path(f"covs_{UNIVERSE}.npy")              # opcional: float32 (T,N,N), se lee con memmap
BENCHMARK_FILE = Path("benchmark_spy.csv")

# --- Datos ------------------------------------------------------------------
START_DATE = "2023-01-01"
END_DATE = "2026-01-01"
WINDOW = 252                        # ventana de estimación de Σ (días)
MAX_NAN_FRACTION = 0.05             # activos con más NaN que esto se descartan

# --- Solver -----------------------------------------------------------------
# "structured": elimina u analíticamente y resuelve un sistema N x N (pdipm_portfolio.py).
# "generic"   : PrimalDualQPSolver con KKT densa (2N+1); correcto pero ~10x más lento en N=500.
SOLVER = "structured"
SLSQP_MAX_N = 150                   # benchmark.py omite SLSQP por encima de este N (es inviable)

# --- Simulación -------------------------------------------------------------
TURNOVER_BUDGET = 0.05              # 0.5*||Δw||_1 máximo por rebalanceo
COST_RATE = 5e-4                    # 5 bps por unidad de nocional negociado (por lado)
RISK_FREE_ANNUAL = 0.04
TRADING_DAYS = 252
L_GRID = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
MAX_FAIL_FRAC = 0.02                # el backtest aborta si el solver falla en más de esta fracción de días
