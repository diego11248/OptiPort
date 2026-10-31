"""
dataset.py
==========
Descarga precios, construye retornos y (opcionalmente) un caché de covarianzas
Ledoit-Wolf rodantes.

    python dataset.py                       # universo de config.UNIVERSE (sp500 por defecto)
    python dataset.py --universe sp100      # tu lista de 100 activos
    python dataset.py --refresh-universe    # vuelve a bajar los constituyentes del S&P 500
    python dataset.py --universe-file mis_tickers.csv   # CSV con columnas Symbol[,Security,GICS Sector]
    python dataset.py --no-cov-cache        # no escribir el caché de Σ (se calcula al vuelo)

Cambios clave para N≈500
------------------------
* NO se guarda un array (T, N, N) en float64: con N=500 y T≈500 son ~1 GB.
  Se guarda un caché OPCIONAL en float32 (~500 MB, leído con memmap) y el motor
  de backtest puede además recalcular Σ_t al vuelo (≈10-30 ms por día).
* `rolling_data_<universo>.npz` es ahora un archivo de metadatos liviano
  (fechas, tickers, ventana, intensidades de shrinkage).
* Convención temporal sin cambios: Σ para dates[t] usa la ventana [t-W, t).

Limitaciones que conviene conocer
---------------------------------
* Los constituyentes son los ACTUALES del índice: sesgo de supervivencia (y de
  selección por pertenencia futura) en cualquier backtest histórico.
* Las empresas con historia corta (salidas a bolsa recientes) se descartan por el
  filtro de NaN, así que N final suele ser algo menor que 500.
"""

from __future__ import annotations

import argparse
import io
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.covariance import ledoit_wolf

import config

FloatArray = NDArray[np.float64]
BENCHMARK = "SPY"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

TICKERS_100: List[str] = [
    # Tecnología de la Información (15)
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'CSCO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'ACN',
    'INTC', 'QCOM', 'TXN', 'IBM', 'AMAT',
    # Servicios de Comunicación (8)
    'GOOGL', 'META', 'NFLX', 'TMUS', 'DIS', 'CMCSA', 'VZ', 'T',
    # Consumo Discrecional (10)
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'BKNG', 'TJX', 'TGT',
    # Consumo Básico (10)
    'PG', 'KO', 'WMT', 'PEP', 'COST', 'PM', 'MDLZ', 'MO', 'CL', 'KMB',
    # Finanzas (15)
    'JPM', 'BAC', 'GS', 'V', 'MA', 'MS', 'WFC', 'C', 'BLK', 'AXP',
    'SCHW', 'SPGI', 'PGR', 'CB', 'GDS',
    # Cuidado de la Salud (12)
    'JNJ', 'UNH', 'PFE', 'ABBV', 'LLY', 'MRK', 'AMGN', 'TMO', 'ABT', 'DHR',
    'BMY', 'CVS',
    # Industriales (12)
    'CAT', 'GE', 'HON', 'UNP', 'BA', 'DE', 'RTX', 'LMT', 'UPS', 'ADP',
    'ETN', 'WM',
    # Energía (6)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC',
    # Servicios Públicos / Utilities (4)
    'NEE', 'DUK', 'SO', 'D',
    # Bienes Raíces / REITs (4)
    'PLD', 'AMT', 'CCI', 'EQIX',
    # Materiales (4)
    'LIN', 'APD', 'SHW', 'FCX',
]


# --------------------------------------------------------------------------- #
# Universo
# --------------------------------------------------------------------------- #
def fetch_sp500_constituents() -> pd.DataFrame:
    """Constituyentes actuales del S&P 500 desde Wikipedia (Symbol, Security, GICS Sector).

    Requiere `requests` y `lxml` (pip install requests lxml). Los símbolos con punto
    (BRK.B, BF.B) se convierten al formato de Yahoo (BRK-B, BF-B).
    """
    import requests

    resp = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0 (portfolio-research)"}, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text), match="Symbol")[0]
    uni = table[["Symbol", "Security", "GICS Sector"]].copy()
    uni["Symbol"] = uni["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    return uni.drop_duplicates("Symbol").reset_index(drop=True)


def load_universe(name: str, universe_file: Optional[str], refresh: bool) -> pd.DataFrame:
    """Devuelve un DataFrame con al menos la columna Symbol (y Security/GICS Sector si existen)."""
    if universe_file:
        uni = pd.read_csv(universe_file)
        if "Symbol" not in uni.columns:
            raise ValueError(f"{universe_file} debe tener una columna 'Symbol'.")
    elif name == "sp100":
        uni = pd.DataFrame({"Symbol": TICKERS_100})
    elif name == "sp500":
        if config.UNIVERSE_FILE.exists() and not refresh:
            uni = pd.read_csv(config.UNIVERSE_FILE)
        else:
            try:
                uni = fetch_sp500_constituents()
            except Exception as exc:  # red, parser HTML, cambio de formato de la página...
                if config.UNIVERSE_FILE.exists():
                    print(f"Aviso: no se pudo actualizar la lista ({exc}); se usa {config.UNIVERSE_FILE}.")
                    uni = pd.read_csv(config.UNIVERSE_FILE)
                else:
                    raise RuntimeError(
                        f"No se pudo descargar la lista del S&P 500 ({exc}). Crea un CSV con "
                        "columnas Symbol,Security,GICS Sector y pásalo con --universe-file.") from exc
    else:
        raise ValueError(f"Universo desconocido: {name!r}.")
    uni = uni.drop_duplicates("Symbol").reset_index(drop=True)
    if BENCHMARK in set(uni["Symbol"]):
        uni = uni[uni["Symbol"] != BENCHMARK]
    return uni


# --------------------------------------------------------------------------- #
# Descarga y limpieza
# --------------------------------------------------------------------------- #
def download_prices(symbols: Sequence[str], start: str, end: str,
                    batch_size: int = 100, retries: int = 3) -> pd.DataFrame:
    """Precios de cierre ajustados, descargados por lotes (500 símbolos de golpe suelen fallar)."""
    import yfinance as yf

    frames: List[pd.DataFrame] = []
    for i in range(0, len(symbols), batch_size):
        batch = list(symbols[i:i + batch_size])
        for attempt in range(1, retries + 1):
            raw = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False,
                              threads=True)
            if not raw.empty:
                close = raw["Close"]
                if isinstance(close, pd.Series):          # un solo símbolo en el lote
                    close = close.to_frame(batch[0])
                frames.append(close)
                break
            print(f"  lote {i // batch_size + 1}: vacío (intento {attempt}/{retries}); reintentando...")
            time.sleep(2.0 * attempt)
        else:
            print(f"  Aviso: el lote {i // batch_size + 1} no devolvió datos y se omite.")
        print(f"  descargados {min(i + batch_size, len(symbols))}/{len(symbols)} símbolos")
    if not frames:
        raise RuntimeError("yfinance no devolvió datos (¿sin conexión?).")
    prices = pd.concat(frames, axis=1)
    return prices.loc[:, ~prices.columns.duplicated()]


def clean_prices(raw: pd.DataFrame, max_nan_fraction: float = config.MAX_NAN_FRACTION) -> pd.DataFrame:
    """Descarta activos con >5% de NaN; solo ffill (NO bfill: copiaría precios futuros hacia atrás).

    Los activos con NaN iniciales remanentes (IPOs recientes) se descartan en vez de inventar retornos.
    """
    keep = raw.columns[raw.isnull().mean() < max_nan_fraction]
    clean = raw[keep].ffill()
    return clean.loc[:, clean.notna().all()]


# --------------------------------------------------------------------------- #
# Covarianzas rodantes (Ledoit-Wolf)
# --------------------------------------------------------------------------- #
def build_cov_cache(returns: FloatArray, window: int, path: Optional[str]) -> FloatArray:
    """Calcula Σ_t para t = 0..T-1 con ventana [t, t+window) de `returns` (fila 0 = primer día).

    Si `path` no es None escribe un .npy float32 (T, N, N) fila a fila (memoria O(N²));
    devuelve las intensidades de shrinkage.
    """
    T_total, N = returns.shape
    T = T_total - window
    if T < 2:
        raise ValueError(f"Se necesitan al menos {window + 2} observaciones; hay {T_total}.")
    mm = None
    if path is not None:
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(T, N, N))
    intens = np.empty(T)
    for k in range(T):
        cov, intens[k] = ledoit_wolf(returns[k:k + window])
        if mm is not None:
            mm[k] = cov
        if (k + 1) % 100 == 0 or k + 1 == T:
            print(f"  covarianzas: {k + 1}/{T}")
    if mm is not None:
        mm.flush()
        del mm
    return intens


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", default=config.UNIVERSE, choices=["sp500", "sp100"])
    ap.add_argument("--universe-file", default=None)
    ap.add_argument("--refresh-universe", action="store_true")
    ap.add_argument("--start", default=config.START_DATE)
    ap.add_argument("--end", default=config.END_DATE)
    ap.add_argument("--no-cov-cache", action="store_true")
    args = ap.parse_args(argv)
    if args.universe != config.UNIVERSE:
        raise SystemExit(f"config.UNIVERSE = {config.UNIVERSE!r} define los nombres de archivo; "
                         f"cámbialo a {args.universe!r} en config.py para generar ese universo.")

    uni = load_universe(args.universe, args.universe_file, args.refresh_universe)
    symbols: List[str] = uni["Symbol"].tolist()
    print(f"Universo {config.UNIVERSE_LABEL}: {len(symbols)} símbolos solicitados + {BENCHMARK}.")

    raw = download_prices(symbols + [BENCHMARK], args.start, args.end)
    clean = clean_prices(raw)
    if BENCHMARK not in clean.columns:
        raise RuntimeError("El benchmark no tiene datos completos.")

    rets = clean.pct_change().dropna()
    tickers = [t for t in symbols if t in rets.columns]
    asset_returns, bench = rets[tickers], rets[BENCHMARK]
    N = len(tickers)
    dropped = sorted(set(symbols) - set(tickers))
    print(f"Activos retenidos: {N}/{len(symbols)}; {asset_returns.shape[0]} días. "
          f"Descartados: {len(dropped)}" + (f" (p. ej. {', '.join(dropped[:8])})" if dropped else ""))
    if asset_returns.shape[0] < config.WINDOW + 2:
        raise RuntimeError("Historia insuficiente para la ventana de estimación.")
    if N > config.WINDOW:
        print(f"Nota: N={N} > ventana={config.WINDOW}: la covarianza muestral es singular; "
              "el shrinkage de Ledoit-Wolf la vuelve definida positiva.")

    uni[uni["Symbol"].isin(tickers)].to_csv(config.UNIVERSE_FILE, index=False)
    asset_returns.to_csv(config.RETURNS_FILE)
    bench.to_csv(config.BENCHMARK_FILE)

    cache_path = None if args.no_cov_cache else str(config.COV_CACHE_FILE)
    print("Calculando covarianzas Ledoit-Wolf...")
    intens = build_cov_cache(asset_returns.to_numpy(dtype=np.float64), config.WINDOW, cache_path)
    if cache_path is None and config.COV_CACHE_FILE.exists():
        config.COV_CACHE_FILE.unlink()   # evita dejar un caché viejo que no corresponde a estos datos

    dates = np.array([str(d.date()) for d in asset_returns.index[config.WINDOW:]])
    np.savez_compressed(config.ROLLING_DATA_FILE, dates=dates, tickers=np.array(tickers),
                        window=np.array(config.WINDOW), intensities=intens)
    size = f", caché de Σ: {config.COV_CACHE_FILE.stat().st_size / 1e6:.0f} MB" if cache_path else ""
    print(f"Guardado {config.ROLLING_DATA_FILE}: {len(dates)} fechas, N={N}, "
          f"shrinkage medio {intens.mean():.4f}{size}")


if __name__ == "__main__":
    main()
