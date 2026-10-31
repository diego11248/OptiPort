"""
dashboard.py
============
Panel de Control Cuantitativo Interactivo para Optimización Convexa de Portafolios Multi-Período.
Modelo de Mínima Varianza con Restricción de Turnover (Norma L1) y Solver Propio Primal-Dual (PDIPM).

Para ejecutar:
    streamlit run dashboard.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config
from backtest_engine import (
    BacktestResult,
    MarketData,
    compute_metrics,
    load_market_data,
    reference_returns,
    run_backtest,
)

# Evitar advertencias de directorio no escribible de Matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# --------------------------------------------------------------------------- #
# Configuración General de la Página Streamlit
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="MPO & PDIPM | Optimización Convexa de Portafolios",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Estilos CSS Personalizados (Diseño Cuantitativo Institucional)
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
    /* Tipografía y contenedor principal */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1350px;
    }
    
    /* Encabezado principal */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        color: #f8fafc;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 12px;
    }
    .badge-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(37, 99, 235, 0.2);
        border: 1px solid #3b82f6;
        color: #93c5fd;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-success {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid #10b981;
        color: #6ee7b7;
    }
    
    /* Tarjetas de Métricas KPI */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px 16px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.1);
        transition: all 0.2s ease;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #475569 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0px 0px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.92rem;
    }
    
    /* Callout y Cajas informativas */
    .insight-card {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 12px 0px;
        color: #14532d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Mapeo de Sectores GICS (legado de 100 activos + universe_<universo>.csv generado por dataset.py)
# --------------------------------------------------------------------------- #
_LEGACY_SECTOR_MAP = {
    # Tecnología de la Información (15)
    'AAPL': 'Tecnología', 'MSFT': 'Tecnología', 'NVDA': 'Tecnología', 'AVGO': 'Tecnología',
    'CSCO': 'Tecnología', 'ORCL': 'Tecnología', 'ADBE': 'Tecnología', 'CRM': 'Tecnología',
    'AMD': 'Tecnología', 'ACN': 'Tecnología', 'INTC': 'Tecnología', 'QCOM': 'Tecnología',
    'TXN': 'Tecnología', 'IBM': 'Tecnología', 'AMAT': 'Tecnología',
    # Servicios de Comunicación (8)
    'GOOGL': 'Comunicación', 'META': 'Comunicación', 'NFLX': 'Comunicación', 'TMUS': 'Comunicación',
    'DIS': 'Comunicación', 'CMCSA': 'Comunicación', 'VZ': 'Comunicación', 'T': 'Comunicación',
    # Consumo Discrecional (10)
    'AMZN': 'Consumo Discrecional', 'TSLA': 'Consumo Discrecional', 'HD': 'Consumo Discrecional',
    'MCD': 'Consumo Discrecional', 'NKE': 'Consumo Discrecional', 'SBUX': 'Consumo Discrecional',
    'LOW': 'Consumo Discrecional', 'BKNG': 'Consumo Discrecional', 'TJX': 'Consumo Discrecional',
    'TGT': 'Consumo Discrecional',
    # Consumo Básico (10)
    'PG': 'Consumo Básico', 'KO': 'Consumo Básico', 'WMT': 'Consumo Básico', 'PEP': 'Consumo Básico',
    'COST': 'Consumo Básico', 'PM': 'Consumo Básico', 'MDLZ': 'Consumo Básico', 'MO': 'Consumo Básico',
    'CL': 'Consumo Básico', 'KMB': 'Consumo Básico',
    # Finanzas (15)
    'JPM': 'Finanzas', 'BAC': 'Finanzas', 'GS': 'Finanzas', 'V': 'Finanzas', 'MA': 'Finanzas',
    'MS': 'Finanzas', 'WFC': 'Finanzas', 'C': 'Finanzas', 'BLK': 'Finanzas', 'AXP': 'Finanzas',
    'SCHW': 'Finanzas', 'SPGI': 'Finanzas', 'PGR': 'Finanzas', 'CB': 'Finanzas', 'GDS': 'Finanzas',
    # Cuidado de la Salud (12)
    'JNJ': 'Salud', 'UNH': 'Salud', 'PFE': 'Salud', 'ABBV': 'Salud', 'LLY': 'Salud',
    'MRK': 'Salud', 'AMGN': 'Salud', 'TMO': 'Salud', 'ABT': 'Salud', 'DHR': 'Salud',
    'BMY': 'Salud', 'CVS': 'Salud',
    # Industriales (12)
    'CAT': 'Industrial', 'GE': 'Industrial', 'HON': 'Industrial', 'UNP': 'Industrial',
    'BA': 'Industrial', 'DE': 'Industrial', 'RTX': 'Industrial', 'LMT': 'Industrial',
    'UPS': 'Industrial', 'ADP': 'Industrial', 'ETN': 'Industrial', 'WM': 'Industrial',
    # Energía (6)
    'XOM': 'Energía', 'CVX': 'Energía', 'COP': 'Energía', 'SLB': 'Energía', 'EOG': 'Energía', 'MPC': 'Energía',
    # Servicios Públicos / Utilities (4)
    'NEE': 'Servicios Públicos', 'DUK': 'Servicios Públicos', 'SO': 'Servicios Públicos', 'D': 'Servicios Públicos',
    # Bienes Raíces / REITs (4)
    'PLD': 'Bienes Raíces', 'AMT': 'Bienes Raíces', 'CCI': 'Bienes Raíces', 'EQIX': 'Bienes Raíces',
    # Materiales (4)
    'LIN': 'Materiales', 'APD': 'Materiales', 'SHW': 'Materiales', 'FCX': 'Materiales'
}

_GICS_ES = {
    "Information Technology": "Tecnología", "Communication Services": "Comunicación",
    "Consumer Discretionary": "Consumo Discrecional", "Consumer Staples": "Consumo Básico",
    "Financials": "Finanzas", "Health Care": "Salud", "Industrials": "Industrial",
    "Energy": "Energía", "Utilities": "Servicios Públicos", "Real Estate": "Bienes Raíces",
    "Materials": "Materiales",
}


def _load_sector_map() -> dict:
    """Sectores del universo activo; el CSV del universo (S&P 500) manda sobre el mapa legado."""
    sector_map = dict(_LEGACY_SECTOR_MAP)
    if config.UNIVERSE_FILE.exists():
        uni = pd.read_csv(config.UNIVERSE_FILE)
        if "GICS Sector" in uni.columns:
            for sym, sec in zip(uni["Symbol"], uni["GICS Sector"]):
                sector_map[sym] = _GICS_ES.get(sec, sec)
    return sector_map


SECTOR_MAP = _load_sector_map()


# --------------------------------------------------------------------------- #
# Carga e Integración de Datos en Caché
# --------------------------------------------------------------------------- #
@st.cache_data
def get_market_data_cached() -> MarketData:
    """Carga los datos de mercado mediante load_market_data() del proyecto."""
    return load_market_data()


@st.cache_data
def run_simulation_cached(L_value: float, cost_bps: float, init_mode: str) -> Tuple[BacktestResult, pd.DataFrame, pd.DataFrame]:
    """Ejecuta el backtest en horizonte rodante usando el motor oficial."""
    data = get_market_data_cached()
    res = run_backtest(data, turnover_limit=L_value, cost_rate=cost_bps * 1e-4, init=init_mode)
    
    weights_df = pd.DataFrame(res.weights, index=res.dates, columns=res.tickers)
    results_df = pd.DataFrame({
        "GrossReturn": res.gross_returns,
        "NetReturn": res.net_returns,
        "Turnover": res.turnover,
        "Iterations": res.iterations,
        "SolverFailed": res.failed
    }, index=res.dates)
    
    return res, weights_df, results_df


# --------------------------------------------------------------------------- #
# Funciones Analíticas Cuantitativas
# --------------------------------------------------------------------------- #
def compute_extended_metrics(net_returns: pd.Series, bench_returns: pd.Series | None, rf_annual: float = config.RISK_FREE_ANNUAL) -> Dict[str, float]:
    """Calcula un conjunto exhaustivo de métricas de desempeño y riesgo financiero."""
    base_m = compute_metrics(net_returns, rf_annual=rf_annual)
    r = net_returns.dropna().to_numpy()
    
    # Sortino Ratio (con respecto a tasa libre de riesgo diaria)
    rf_daily = rf_annual / config.TRADING_DAYS
    downside_diff = np.minimum(0.0, r - rf_daily)
    downside_dev = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(config.TRADING_DAYS)
    sortino = (base_m["ann_return"] - rf_annual) / downside_dev if downside_dev > 0 else np.nan
    
    # Calmar Ratio
    calmar = base_m["ann_return"] / abs(base_m["max_drawdown"]) if base_m["max_drawdown"] != 0 else np.nan
    
    # Value at Risk (VaR 95%) y Conditional VaR (CVaR / Expected Shortfall 95%)
    var_95 = -float(np.percentile(r, 5))
    cvar_95 = -float(r[r <= -var_95].mean()) if len(r[r <= -var_95]) > 0 else var_95
    
    # Métricas contra el Benchmark (Beta, Alpha, Information Ratio)
    beta, alpha, ir = np.nan, np.nan, np.nan
    if bench_returns is not None:
        rb = bench_returns.reindex(net_returns.index).dropna().to_numpy()
        if len(rb) == len(r) and len(rb) > 2:
            cov_mat = np.cov(r, rb)
            var_b = cov_mat[1, 1]
            if var_b > 0:
                beta = float(cov_mat[0, 1] / var_b)
                bench_ann = float(rb.mean() * config.TRADING_DAYS)
                alpha = float(base_m["ann_return"] - (rf_annual + beta * (bench_ann - rf_annual)))
                
                diff = r - rb
                tracking_error = float(diff.std(ddof=1) * np.sqrt(config.TRADING_DAYS))
                if tracking_error > 0:
                    ir = float(diff.mean() * config.TRADING_DAYS / tracking_error)
                    
    return {
        **base_m,
        "sortino": sortino,
        "calmar": calmar,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "beta": beta,
        "alpha": alpha,
        "information_ratio": ir
    }


def compute_monthly_tearsheet(daily_returns: pd.Series) -> pd.DataFrame:
    """Genera la matriz clásica de retornos mensuales y anuales (Hedge Fund Tearsheet)."""
    monthly = (1.0 + daily_returns).resample("ME").prod() - 1.0
    df = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values
    })
    table = df.pivot(index="Year", columns="Month", values="Return")
    month_names = {
        1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
    }
    table = table.rename(columns=month_names)
    
    # Retorno Anual Compuesto
    annual = (1.0 + daily_returns).resample("YE").prod() - 1.0
    table["Año Total"] = annual.values
    return table


# --------------------------------------------------------------------------- #
# Barra Lateral: Controles y Perfiles de Simulación
# --------------------------------------------------------------------------- #
st.sidebar.markdown("### Configuración del Modelo")

# Verificación de datos de entrada
if not config.ROLLING_DATA_FILE.exists() or not config.RETURNS_FILE.exists():
    st.error(f"No se encontraron los archivos de datos `{config.ROLLING_DATA_FILE}` o `{config.RETURNS_FILE}`.")
    st.info("Ejecuta primero `python dataset.py` en la terminal para descargar y preparar los datos de mercado.")
    st.stop()

# Cargar metadatos
market_data = get_market_data_cached()
min_date = market_data.dates[0].date()
max_date = market_data.dates[-1].date()

# Presets de Escenarios
preset = st.sidebar.selectbox(
    "Perfil de Estrategia:",
    options=[
        "Institucional Equilibrado (L = 0.05, 5 bps)",
        "Baja Rotación / Conservador (L = 0.01, 5 bps)",
        "Rotación Flexible (L = 0.15, 5 bps)",
        "Teórico Sin Límite (L = 1.00, 5 bps)",
        "Personalizado"
    ],
    index=0
)

# Asignar valores por defecto según el preset
if preset.startswith("Institucional"):
    def_L, def_cost, def_init = 0.05, 5.0, "minvar"
elif preset.startswith("Baja"):
    def_L, def_cost, def_init = 0.01, 5.0, "minvar"
elif preset.startswith("Rotación"):
    def_L, def_cost, def_init = 0.15, 5.0, "minvar"
elif preset.startswith("Teórico"):
    def_L, def_cost, def_init = 1.00, 5.0, "minvar"
else:
    def_L, def_cost, def_init = float(config.TURNOVER_BUDGET), float(config.COST_RATE * 1e4), "minvar"

L_param = st.sidebar.slider(
    "Presupuesto de Turnover ($L$):",
    min_value=0.01, max_value=1.00,
    value=def_L,
    step=0.01,
    help="Límite máximo de rotación unidireccional por día: 0.5 * ||Δw||_1 <= L"
)

cost_param = st.sidebar.number_input(
    "Costo de Transacción (bps por lado):",
    min_value=0.0, max_value=50.0,
    value=def_cost,
    step=0.5,
    help="Fricción lineal cobrada sobre el nocional total transaccionado (1 bps = 0.01%)"
)

init_param = st.sidebar.selectbox(
    "Estrategia de Inicialización (Día 0):",
    options=["minvar", "equal"],
    index=0 if def_init == "minvar" else 1,
    format_func=lambda x: "Mínima Varianza Unconstrained (minvar)" if x == "minvar" else "1/N Iguales Pesos (equal)",
    help="Punto de partida del portafolio en el primer día de simulación"
)

rf_param = st.sidebar.number_input(
    "Tasa Libre de Riesgo Anual ($R_f$):",
    min_value=0.0, max_value=0.15,
    value=float(config.RISK_FREE_ANNUAL),
    step=0.005,
    format="%.3f",
    help="Utilizada para el cálculo de los ratios Sharpe y Sortino y el Alpha de Jensen"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Ventana de Análisis")
date_range = st.sidebar.date_input(
    "Rango de Fechas:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Filtra visualizaciones y métricas para una ventana temporal específica"
)

# Resolver fechas seleccionadas
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_filter, end_filter = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_filter, end_filter = pd.to_datetime(min_date), pd.to_datetime(max_date)

# Ejecución / Recuperación en Caché de la Simulación
with st.spinner("Ejecutando optimización rodante con solver PDIPM..."):
    res_obj, weights_full, results_full = run_simulation_cached(L_param, cost_param, init_param)

# Aplicar filtro de fecha sobre los resultados
mask = (results_full.index >= start_filter) & (results_full.index <= end_filter)
results_df = results_full.loc[mask].copy()
weights_df = weights_full.loc[mask].copy()

if results_df.empty or len(results_df) < 5:
    st.error("El rango de fechas seleccionado contiene muy pocos días para un análisis estadístico fiable.")
    st.stop()

# Benchmark SPY filtrado
spy_series = market_data.benchmark.loc[results_df.index] if market_data.benchmark is not None else None
ref_returns = reference_returns(market_data)
equal_series = pd.Series(ref_returns["1/N diario"], index=market_data.dates).loc[results_df.index]

# --------------------------------------------------------------------------- #
# Encabezado Principal y Badges
# --------------------------------------------------------------------------- #
active_binding_pct = (results_df["Turnover"] >= 0.98 * L_param).mean() * 100
total_days = len(results_df)
n_solver_fail = int(results_df["SolverFailed"].sum())
conv_pct = 100.0 * (1.0 - n_solver_fail / max(total_days - 1, 1))

st.markdown(f"""
<div class="hero-container">
    <p class="hero-title"> Optimización Convexa de Portafolios Multi-Período</p>
    <p class="hero-subtitle">
        Modelo de Mínima Varianza Condicional con Restricción de Turnover en Norma $L_1$ y Solver de Puntos Interiores Primal-Dual (PDIPM)
    </p>
    <div class="badge-container">
        <span class="badge">Universo: {len(market_data.tickers)} Activos ({config.UNIVERSE_LABEL})</span>
        <span class="badge">Presupuesto L: {L_param:.2f} ({L_param*100:.1f}%)</span>
        <span class="badge">Fricción: {cost_param:.1f} bps</span>
        <span class="badge">Horizonte: {start_filter.strftime('%d/%m/%Y')} a {end_filter.strftime('%d/%m/%Y')} ({total_days} días)</span>
        <span class="badge badge-success">Solver: {conv_pct:.1f}% Convergencia ({n_solver_fail} Fallos)</span>
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Métricas Ejecutivas (KPIs Principales)
# --------------------------------------------------------------------------- #
metrics_net = compute_extended_metrics(results_df["NetReturn"], spy_series, rf_annual=rf_param)
metrics_gross = compute_extended_metrics(results_df["GrossReturn"], spy_series, rf_annual=rf_param)
metrics_spy = compute_extended_metrics(spy_series, None, rf_annual=rf_param) if spy_series is not None else {}

kpi_cols = st.columns(6)
kpi_cols[0].metric(
    "Retorno Neto Anual",
    f"{metrics_net['ann_return']:.2%}",
    delta=f"{(metrics_net['ann_return'] - metrics_spy.get('ann_return', 0.0)):+.2%} vs SPY" if spy_series is not None else None,
    help="Retorno anualizado aritmético neto de costos de transacción."
)
kpi_cols[1].metric(
    "Volatilidad Anual",
    f"{metrics_net['ann_vol']:.2%}",
    delta=f"{(metrics_net['ann_vol'] - metrics_spy.get('ann_vol', 0.0)):.2%} vs SPY" if spy_series is not None else None,
    delta_color="inverse",
    help="Desviación estándar anualizada (252 días). Menor volatilidad indica mayor estabilidad."
)
kpi_cols[2].metric(
    "Sharpe Ratio Neto",
    f"{metrics_net['sharpe']:.2f}",
    help=f"Ratio Sharpe neto (Rf = {rf_param:.1%}). Error estándar de Lo (2002): ±{metrics_net['sharpe_se']:.2f}"
)
kpi_cols[3].metric(
    "Sortino Ratio",
    f"{metrics_net['sortino']:.2f}",
    help="Mide el retorno excedente sobre la desviación a la baja (downside semivariance)."
)
kpi_cols[4].metric(
    "Max Drawdown",
    f"{metrics_net['max_drawdown']:.2%}",
    delta=f"{(metrics_net['max_drawdown'] - metrics_spy.get('max_drawdown', 0.0)):.2%} vs SPY" if spy_series is not None else None,
    delta_color="off",
    help="Máxima caída acumulada desde el pico histórico de riqueza."
)
kpi_cols[5].metric(
    "Beta vs SPY",
    f"{metrics_net['beta']:.2f}" if not np.isnan(metrics_net['beta']) else "N/A",
    help="Sensibilidad al riesgo sistemático de mercado. Un Beta < 1.0 refleja carácter defensivo."
)

st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Pestañas Interactivas Principales
# --------------------------------------------------------------------------- #
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Rendimiento & Riesgo",
    "💼 Asignación & Sectores",
    "🔄 Turnover & Fricción",
    "🔬 Análisis de Sensibilidad",
    "⚡ Diagnóstico del Solver",
    "📐 Formulación Matemática"
])


# =========================================================================== #
# TAB 1: RENDIMIENTO FINANCIERO Y ANÁLISIS DE RIESGO
# =========================================================================== #
with tab1:
    st.subheader("Curva de Crecimiento del Capital (Equity Curve)")
    
    # Construcción de curvas de riqueza acumulada
    cum_net = np.cumprod(1.0 + results_df["NetReturn"]) - 1.0
    cum_gross = np.cumprod(1.0 + results_df["GrossReturn"]) - 1.0
    
    chart_data = pd.DataFrame({
        "MPO Neto (con costos)": cum_net,
        "MPO Bruto (sin costos)": cum_gross,
    }, index=results_df.index)
    
    if spy_series is not None:
        chart_data["Benchmark SPY"] = np.cumprod(1.0 + spy_series) - 1.0
    chart_data["Benchmark 1/N"] = np.cumprod(1.0 + equal_series) - 1.0
    
    # Gráfica interactiva de Streamlit
    st.line_chart(chart_data, color=["#1E3A8A", "#94A3B8", "#D97706", "#10B981"], height=380)
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.subheader("Perfil de Drawdown Submarino")
        # Serie de drawdown
        wealth_net = np.cumprod(1.0 + results_df["NetReturn"])
        peaks_net = np.maximum.accumulate(wealth_net)
        dd_net = (wealth_net - peaks_net) / peaks_net
        
        dd_df = pd.DataFrame({"Drawdown MPO Neto": dd_net}, index=results_df.index)
        if spy_series is not None:
            wealth_spy = np.cumprod(1.0 + spy_series)
            peaks_spy = np.maximum.accumulate(wealth_spy)
            dd_df["Drawdown SPY"] = (wealth_spy - peaks_spy) / peaks_spy
            
        st.line_chart(dd_df, color=["#DC2626", "#D97706"], height=260)
        st.caption("Caída porcentual acumulada desde máximos históricos precedentes.")
        
    with col_t2:
        st.subheader("Riesgo Móvil Rodante (Ventana 60 Días)")
        rolling_vol_net = results_df["NetReturn"].rolling(60).std() * np.sqrt(config.TRADING_DAYS)
        rvol_df = pd.DataFrame({"Volatilidad MPO Neto": rolling_vol_net}, index=results_df.index)
        if spy_series is not None:
            rvol_df["Volatilidad SPY"] = spy_series.rolling(60).std() * np.sqrt(config.TRADING_DAYS)
            
        st.line_chart(rvol_df, color=["#1E3A8A", "#D97706"], height=260)
        st.caption("Volatilidad anualizada rodante de 60 días de negociación.")

    st.markdown("---")
    
    # Tearsheet Mensual
    st.subheader("Matriz de Rendimientos Mensuales (Tearsheet)")
    monthly_table = compute_monthly_tearsheet(results_df["NetReturn"])
    
    styled_monthly = monthly_table.style.format(
        "{:+.2%}", na_rep="-"
    ).background_gradient(
        cmap="RdYlGn", vmin=-0.08, vmax=0.08, subset=[c for c in monthly_table.columns if c != "Año Total"]
    ).background_gradient(
        cmap="Blues", subset=["Año Total"]
    )
    st.dataframe(styled_monthly, use_container_width=True)

    # Tabla Resumen Comparativa
    st.subheader("Tabla Resumen Cuantitativa Multiactivo")
    comp_data = {
        "MPO Neto (Estrategia)": metrics_net,
        "MPO Bruto (Sin fricción)": metrics_gross,
    }
    if spy_series is not None:
        comp_data["Benchmark SPY"] = metrics_spy
    comp_data["Benchmark 1/N"] = compute_extended_metrics(equal_series, spy_series, rf_annual=rf_param)
    
    comp_df = pd.DataFrame(comp_data).T[[
        "ann_return", "ann_return_geo", "ann_vol", "sharpe", "sortino", "calmar", "max_drawdown", "var_95", "cvar_95", "beta", "alpha"
    ]].rename(columns={
        "ann_return": "Retorno Anual (Arit.)",
        "ann_return_geo": "Retorno Anual (Geom.)",
        "ann_vol": "Volatilidad Anual",
        "sharpe": "Sharpe Ratio",
        "sortino": "Sortino Ratio",
        "calmar": "Calmar Ratio",
        "max_drawdown": "Max Drawdown",
        "var_95": "VaR Diario (95%)",
        "cvar_95": "CVaR Diario (95%)",
        "beta": "Beta vs SPY",
        "alpha": "Alpha Jensen"
    })
    
    format_dict = {
        "Retorno Anual (Arit.)": "{:.2%}",
        "Retorno Anual (Geom.)": "{:.2%}",
        "Volatilidad Anual": "{:.2%}",
        "Sharpe Ratio": "{:.2f}",
        "Sortino Ratio": "{:.2f}",
        "Calmar Ratio": "{:.2f}",
        "Max Drawdown": "{:.2%}",
        "VaR Diario (95%)": "{:.2%}",
        "CVaR Diario (95%)": "{:.2%}",
        "Beta vs SPY": "{:.2f}",
        "Alpha Jensen": "{:+.2%}"
    }
    st.dataframe(comp_df.style.format(format_dict, na_rep="N/A"), use_container_width=True)


# =========================================================================== #
# TAB 2: ASIGNACIÓN DE ACTIVOS Y EXPOSICIÓN SECTORIAL
# =========================================================================== #
with tab2:
    st.subheader("Descomposición Sectorial Dinámica (GICS)")
    st.markdown("Evolución temporal de la exposición agrupada en los 11 sectores económicos oficiales.")
    
    # Agrupación sectorial evitando advertencias de pandas
    sector_series = pd.Series(SECTOR_MAP).reindex(weights_df.columns).fillna("Otro")
    sector_weights = weights_df.T.groupby(sector_series).sum().T
    
    # Gráfico de área apilada interactivo
    st.area_chart(sector_weights, height=360)
    
    col_w1, col_w2 = st.columns([1.2, 1])
    
    with col_w1:
        st.subheader("Evolución de Activos Individuales Principales")
        top_k = st.slider("Cantidad de activos principales a desglosar:", min_value=3, max_value=20, value=8)
        
        # Identificar los top_k activos por peso promedio
        top_assets = weights_df.mean().nlargest(top_k).index
        weights_top = weights_df[top_assets].copy()
        weights_top["Otros Activos"] = 1.0 - weights_top.sum(axis=1)
        
        st.area_chart(weights_top, height=320)
        
    with col_w2:
        st.subheader("Instantánea del Portafolio Actual (Último Día)")
        latest_weights = weights_df.iloc[-1].sort_values(ascending=False)
        latest_top10 = latest_weights.head(10).reset_index()
        latest_top10.columns = ["Activo", "Ponderación Actual"]
        latest_top10["Sector"] = latest_top10["Activo"].map(SECTOR_MAP).fillna("Otro")
        
        st.dataframe(
            latest_top10.style.format({"Ponderación Actual": "{:.2%}"}),
            use_container_width=True,
            height=320
        )

    st.markdown("---")
    
    # Concentración y Diversificación
    st.subheader("Métricas de Concentración y Diversificación")
    col_div1, col_div2 = st.columns([2, 1])
    
    with col_div1:
        # Número Efectivo de Activos: N_eff = 1 / sum(w_i^2)
        hhi = (weights_df ** 2).sum(axis=1)
        n_eff = 1.0 / hhi
        
        div_df = pd.DataFrame({
            "Número Efectivo de Activos (N_eff)": n_eff,
            "Índice HHI (Concentración)": hhi * 100
        }, index=weights_df.index)
        
        st.line_chart(div_df["Número Efectivo de Activos (N_eff)"], color=["#2563EB"], height=250)
        st.caption("Número efectivo de constituyentes independientes $N_{\\text{eff}} = 1 / \\sum w_i^2$. Un valor mayor indica mayor diversificación efectiva.")
        
    with col_div2:
        st.markdown("#### Estadísticas de Diversificación")
        st.write(f"- **$N_{{eff}}$ Promedio:** {n_eff.mean():.1f} activos")
        st.write(f"- **$N_{{eff}}$ Mínimo:** {n_eff.min():.1f} activos")
        st.write(f"- **$N_{{eff}}$ Máximo:** {n_eff.max():.1f} activos")
        st.write(f"- **HHI Promedio:** {hhi.mean():.4f}")
        st.info("💡 En un portafolio de mínima varianza sin restricciones sectoriales, el capital se concentra de forma natural en sectores de baja covarianza (Consumo Básico, Utilities, Salud).")

    # Explorador Completo de Activos
    st.subheader(f"Explorador Exhaustivo de los {len(market_data.tickers)} Activos")
    asset_summary = pd.DataFrame({
        "Sector": pd.Series(SECTOR_MAP).reindex(weights_df.columns).fillna("Otro"),
        "Peso Promedio": weights_df.mean(),
        "Peso Máximo": weights_df.max(),
        "Peso Actual": weights_df.iloc[-1],
        "Volatilidad Diaria": pd.DataFrame(market_data.returns, index=market_data.dates, columns=market_data.tickers).loc[weights_df.index].std() * np.sqrt(config.TRADING_DAYS)
    }).sort_values(by="Peso Promedio", ascending=False)
    
    st.dataframe(
        asset_summary.style.format({
            "Peso Promedio": "{:.2%}",
            "Peso Máximo": "{:.2%}",
            "Peso Actual": "{:.2%}",
            "Volatilidad Diaria": "{:.2%}"
        }),
        use_container_width=True,
        height=350
    )


# =========================================================================== #
# TAB 3: TURNOVER, COSTOS Y DINÁMICA DE RESTRICCIONES
# =========================================================================== #
with tab3:
    st.subheader("Dinámica del Presupuesto de Turnover ($L_1$)")
    
    col_to1, col_to2 = st.columns([2, 1])
    
    with col_to1:
        to_plot_df = pd.DataFrame({
            "Turnover Diario Observado": results_df["Turnover"] * 100,
            "Presupuesto Máximo (L)": L_param * 100
        }, index=results_df.index)
        
        st.line_chart(to_plot_df, color=["#EA580C", "#DC2626"], height=300)
        st.caption("Consumo diario del presupuesto de rotación $0.5 \\|w_{new} - w_{drift}\\|_1$ comparado contra la cota superior $L$.")
        
    with col_to2:
        st.markdown("#### Análisis de Saturación KKT")
        binding_days = int((results_df["Turnover"] >= 0.98 * L_param).sum())
        st.metric("Tasa de Restricción Activa", f"{active_binding_pct:.1f}%")
        st.write(f"- **Días con Cota Saturada:** {binding_days} / {total_days}")
        st.write(f"- **Turnover Diario Medio:** {results_df['Turnover'].mean():.2%}")
        st.write(f"- **Turnover Diario Máximo:** {results_df['Turnover'].max():.2%}")
        st.write(f"- **Turnover Anualizado Est.:** {results_df['Turnover'].mean() * config.TRADING_DAYS:.1f}x")

    st.markdown("---")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.subheader("Arrastre de Costos de Transacción (Cost Drag)")
        wealth_gross = np.cumprod(1.0 + results_df["GrossReturn"])
        wealth_net = np.cumprod(1.0 + results_df["NetReturn"])
        cost_drag = (wealth_gross - wealth_net) * 100
        
        drag_df = pd.DataFrame({"Pérdida Acumulada de Capital (p.p.)": cost_drag}, index=results_df.index)
        st.line_chart(drag_df, color=["#DC2626"], height=250)
        st.caption("Diferencia porcentual entre el capital bruto acumulado y el capital neto tras pagar costos de fricción.")
        
    with col_c2:
        st.subheader("Distribución de Frecuencia de Rebalanceos")
        fig_hist, ax_hist = plt.subplots(figsize=(6, 3.8))
        ax_hist.hist(results_df["Turnover"] * 100, bins=25, color="#F97316", edgecolor="#C2410C", alpha=0.85)
        ax_hist.axvline(results_df["Turnover"].mean() * 100, color="navy", linestyle="--", lw=1.5, label=f"Media: {results_df['Turnover'].mean()*100:.2f}%")
        ax_hist.axvline(L_param * 100, color="red", linestyle=":", lw=1.5, label=f"Límite L: {L_param*100:.1f}%")
        ax_hist.set_xlabel("Turnover Diario (%)")
        ax_hist.set_ylabel("Frecuencia (Días)")
        ax_hist.legend(loc="upper right", fontsize=8)
        ax_hist.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_hist)


# =========================================================================== #
# TAB 4: ANÁLISIS DE SENSIBILIDAD DEL PARÁMETRO L
# =========================================================================== #
with tab4:
    st.subheader("Frontera de Fricción: Barrido del Presupuesto de Turnover ($L$)")
    st.markdown("""
    Evaluación empírica del trade-off entre **adaptabilidad a la estructura de covarianza rodante** y **costos de transacción acumulados**.
    """)
    
    sens_file = Path("sensibilidad_resultados.csv")
    if sens_file.exists():
        sens_df = pd.read_csv(sens_file)
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("#### Trade-off de Pareto: Sharpe Neto vs. Turnover Diario")
            # Gráfica de Pareto con barras de error estándar
            sens_df["ymin"] = sens_df["Sharpe neto"] - sens_df["Sharpe SE"]
            sens_df["ymax"] = sens_df["Sharpe neto"] + sens_df["Sharpe SE"]
            sens_df["label"] = ["L=" + str(l) for l in sens_df["L"]]
            
            base_chart = alt.Chart(sens_df).encode(
                x=alt.X("Turnover diario (%):Q", title="Turnover Diario Promedio (%)")
            )
            error_bars = base_chart.mark_rule(color="#94a3b8").encode(
                y=alt.Y("ymin:Q", title="Sharpe Ratio Neto"),
                y2="ymax:Q"
            )
            points = base_chart.mark_circle(size=90, color="#2563EB").encode(
                y=alt.Y("Sharpe neto:Q"),
                tooltip=["L", "Sharpe neto", "Sharpe SE", "Turnover diario (%)", "Retorno neto anual (%)", "Volatilidad anual (%)"]
            )
            text_labels = base_chart.mark_text(dy=-12, color="#0f172a", fontWeight="bold").encode(
                y=alt.Y("Sharpe neto:Q"),
                text=alt.Text("label:N")
            )
            st.altair_chart((error_bars + points + text_labels).properties(height=320), use_container_width=True)
            st.caption("Barras de error corresponden a ±1 error estándar asintótico (Lo, 2002).")
            
        with col_s2:
            st.markdown("#### Frontera Retorno Neto vs. Volatilidad")
            fig_front, ax_front = plt.subplots(figsize=(6, 4))
            scatter = ax_front.scatter(
                sens_df["Volatilidad anual (%)"],
                sens_df["Retorno neto anual (%)"],
                c=sens_df["L"],
                cmap="viridis",
                s=120,
                edgecolors="black"
            )
            for _, r in sens_df.iterrows():
                ax_front.annotate(f" L={r['L']}", (r["Volatilidad anual (%)"], r["Retorno neto anual (%)"]), fontsize=8)
            ax_front.set_xlabel("Volatilidad Anual (%)")
            ax_front.set_ylabel("Retorno Neto Anual (%)")
            ax_front.grid(True, linestyle="--", alpha=0.5)
            cbar = plt.colorbar(scatter, ax=ax_front)
            cbar.set_label("Presupuesto L", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_front)
            
        st.markdown("#### Tabla Cuantitativa de Sensibilidad Sistemática")
        st.dataframe(
            sens_df[[
                "L", "Retorno neto anual (%)", "Volatilidad anual (%)", "Sharpe neto",
                "Sharpe SE", "Max drawdown (%)", "Turnover diario (%)", "Iter. solver", "Fallos"
            ]].style.format({
                "L": "{:.2f}",
                "Retorno neto anual (%)": "{:.2f}%",
                "Volatilidad anual (%)": "{:.2f}%",
                "Sharpe neto": "{:.3f}",
                "Sharpe SE": "{:.3f}",
                "Max drawdown (%)": "{:.2f}%",
                "Turnover diario (%)": "{:.2f}%",
                "Iter. solver": "{:.1f}",
                "Fallos": "{:.0f}"
            }),
            use_container_width=True
        )
        
        best = sens_df.loc[sens_df["Sharpe neto"].idxmax()]
        sens_years = total_days / config.TRADING_DAYS
        within_1se = sens_df[sens_df["Sharpe neto"] >= best["Sharpe neto"] - best["Sharpe SE"]]["L"].tolist()
        st.markdown(f"""
        <div class="insight-card">
            <b>💡 Lectura de la sensibilidad (calculada con la última corrida):</b><br>
            • <b>Mayor Sharpe neto:</b> L = {best['L']:.2f} ({best['Sharpe neto']:.2f} ± {best['Sharpe SE']:.2f}).<br>
            • <b>Significancia:</b> con ~{sens_years:.1f} años de datos, los niveles L = {', '.join(f'{v:.2f}' for v in within_1se)} están a menos de 1 error estándar del mejor; sus diferencias no son estadísticamente distinguibles.<br>
            • <b>Fricción:</b> el turnover diario va de {sens_df['Turnover diario (%)'].min():.2f}% a {sens_df['Turnover diario (%)'].max():.2f}% entre los extremos de la grilla.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Para generar el barrido sistemático de sensibilidad, ejecuta en la terminal: `python sensitivity.py`.")


# =========================================================================== #
# TAB 5: DIAGNÓSTICO NUMÉRICO Y BENCHMARK DEL SOLVER (PDIPM)
# =========================================================================== #
with tab5:
    st.subheader("Análisis de Desempeño Numérico del Solver Primal-Dual (PDIPM)")
    
    bench_file = Path("benchmark_3way_summary.csv")
    if bench_file.exists():
        bench_df = pd.read_csv(bench_file)
        st.markdown("#### Benchmark Comparativo Multidependencia: PDIPM vs. CVXPY (Clarabel) vs. SLSQP (SciPy)")
        st.table(bench_df)
        
        num_file = Path("benchmark_3way_numeric.csv")
        if not num_file.exists():
            st.info("Ejecuta `python benchmark.py` para generar las cifras numéricas del benchmark "
                    "(benchmark_3way_numeric.csv); las gráficas se calculan a partir de ese archivo.")
        else:
            num_df = pd.read_csv(num_file)
            bench_n, bench_days = int(num_df["N"].iloc[0]), int(num_df["Días"].iloc[0])
            palette = ["#2563EB", "#10B981", "#EF4444"]
            col_bm1, col_bm2 = st.columns(2)

            with col_bm1:
                st.markdown("#### Tiempo de Ejecución por Rebalanceo")
                fig_time, ax_time = plt.subplots(figsize=(6, 3.8))
                bars = ax_time.bar(num_df["Solver"], num_df["Tiempo Total (ms)"], color=palette[:len(num_df)])
                ax_time.set_ylabel("Milisegundos (ms)")
                ax_time.set_yscale("log")
                ax_time.set_title(f"Tiempo de Solución Total (log) — N={bench_n}")
                ax_time.tick_params(axis="x", labelsize=7)
                for bar in bars:
                    yval = bar.get_height()
                    ax_time.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.15, f"{yval:.1f} ms",
                                 ha="center", va="bottom", fontsize=8)
                ax_time.grid(True, linestyle="--", alpha=0.4, axis="y")
                plt.tight_layout()
                st.pyplot(fig_time)

            with col_bm2:
                st.markdown("#### Confiabilidad Numérica y Tasa de Fallos")
                fig_fail, ax_fail = plt.subplots(figsize=(6, 3.8))
                bars_f = ax_fail.bar(num_df["Solver"], num_df["Tasa de Convergencia (%)"], color=palette[:len(num_df)])
                ax_fail.set_ylabel("Tasa de Convergencia (%)")
                ax_fail.set_ylim(0, 115)
                ax_fail.set_title(f"Robustez y Factibilidad en {bench_days} Días")
                ax_fail.tick_params(axis="x", labelsize=7)
                for bar in bars_f:
                    yval = bar.get_height()
                    ax_fail.text(bar.get_x() + bar.get_width() / 2.0, yval + 2, f"{yval:.1f}%",
                                 ha="center", va="bottom", fontsize=8)
                ax_fail.grid(True, linestyle="--", alpha=0.4, axis="y")
                plt.tight_layout()
                st.pyplot(fig_fail)

            own = num_df.iloc[0]
            cvx_rows = num_df[num_df["Solver"].str.contains("CVXPY")]
            lines = [f"• <b>Corrida medida:</b> N={bench_n} activos, {bench_days} rebalanceos; el PDIPM propio tarda "
                     f"{own['Tiempo Total (ms)']:.1f} ms por rebalanceo ({own['Tasa de Convergencia (%)']:.1f}% de convergencia)."]
            if not cvx_rows.empty:
                cvx = cvx_rows.iloc[0]
                overhead = cvx["Tiempo Total (ms)"] - cvx["Tiempo Solver Puro (ms)"]
                lines.append(f"• <b>CVXPY (Clarabel):</b> {cvx['Tiempo Solver Puro (ms)']:.1f} ms de solver puro más "
                             f"{overhead:.1f} ms de compilación del DSL ({cvx['Tiempo Total (ms)']:.1f} ms en total).")
            slsqp_rows = num_df[num_df["Solver"].str.contains("SLSQP")]
            if not slsqp_rows.empty:
                sl = slsqp_rows.iloc[0]
                lines.append(f"• <b>SLSQP:</b> {sl['Tasa de Convergencia (%)']:.1f}% de convergencia "
                             f"({int(sl['Fallos'])}/{bench_days} fallos) y {sl['Tiempo Total (ms)']:.0f} ms por rebalanceo.")
            else:
                lines.append(f"• <b>SLSQP:</b> omitido en esta corrida (inviable con N={bench_n}).")
            st.markdown('<div class="insight-card"><b>⚡ Hallazgos del Benchmark de Solvers:</b><br>'
                        + "<br>".join(lines) + "</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    col_it1, col_it2 = st.columns(2)
    
    with col_it1:
        st.markdown("#### Distribución de Iteraciones Newton (Predictor-Corrector Mehrotra)")
        solved_iters = results_df["Iterations"][results_df["Iterations"] > 0]
        if not solved_iters.empty:
            fig_it, ax_it = plt.subplots(figsize=(6, 3.6))
            ax_it.hist(solved_iters, bins=np.arange(solved_iters.min() - 0.5, solved_iters.max() + 1.5, 1),
                       color="#3B82F6", edgecolor="#1D4ED8", alpha=0.85)
            ax_it.axvline(solved_iters.mean(), color="red", linestyle="--", lw=1.5, label=f"Media: {solved_iters.mean():.1f}")
            ax_it.set_xlabel("Iteraciones de Newton")
            ax_it.set_ylabel("Frecuencia (Días)")
            ax_it.legend(loc="upper right")
            ax_it.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_it)
            
    with col_it2:
        st.markdown("#### Trayectoria Temporal de Iteraciones")
        st.line_chart(results_df["Iterations"], color=["#3B82F6"], height=250)
        st.caption("Consistencia en el número de iteraciones requeridas para alcanzar la tolerancia de convergencia (tol = 1e-8).")


# =========================================================================== #
# TAB 6: FORMULACIÓN MATEMÁTICA Y MODELO DE OPTIMIZACIÓN
# =========================================================================== #
with tab6:
    st.subheader("Formulación Matemática y Arquitectura Algorítmica")
    
    st.markdown(r"""
    ### 1. Problema Primal de Mínima Varianza Multi-Período
    En cada día $t$, el gestor busca la asignación de pesos $w \in \mathbb{R}^N$ que minimice la varianza condicional del portafolio, anclado a los pesos derivados al cierre anterior $w_{\text{drift}}$, sujeto a un presupuesto de rotación $L$:

    $$\min_{w \in \mathbb{R}^N} \quad \frac{1}{2} w^T (2\Sigma_t) w$$

    sujeto a:
    $$\mathbf{1}^T w = 1 \quad \text{(Plena inversión)}$$
    $$w \ge 0 \quad \text{(Restricción Long-Only)}$$
    $$\frac{1}{2} \|w - w_{\text{drift}}\|_1 \le L \quad \text{(Presupuesto de Turnover)}$$

    ---

    ### 2. Reformulación QP Canónica mediante Variables de Holgura
    Para eliminar la no-diferenciabilidad de la norma $L_1$, se introducen variables auxiliares $u \in \mathbb{R}^N$ tales que $|w_i - w_{i,\text{drift}}| \le u_i$:

    $$\min_{x = \begin{bmatrix} w \\ u \end{bmatrix} \in \mathbb{R}^{2N}} \quad \frac{1}{2} x^T \begin{bmatrix} 2\Sigma & 0 \\ 0 & 0 \end{bmatrix} x$$

    sujeto a las restricciones de igualdad y desigualdad lineales:
    $$\begin{bmatrix} \mathbf{1}^T & \mathbf{0}^T \end{bmatrix} x = 1$$
    $$\begin{bmatrix} -I_N & 0 \\ -I_N & -I_N \\ I_N & -I_N \\ \mathbf{0}^T & \mathbf{1}^T \end{bmatrix} \begin{bmatrix} w \\ u \end{bmatrix} \le \begin{bmatrix} \mathbf{0} \\ -w_{\text{drift}} \\ w_{\text{drift}} \\ 2L \end{bmatrix}$$

    ---

    ### 3. Condiciones de Optimalidad KKT y Sistema Aumentado
    El Lagrangiano del problema primal-dual es:
    $$\mathcal{L}(x, y, z) = \frac{1}{2} x^T Q x + y^T (A x - b) + z^T (G x + s - h)$$

    donde $s \ge 0$ son variables de holgura y $z \ge 0$ son multiplicadores duales. El sistema no lineal KKT perturbado por la barrera $\tau > 0$ es:
    $$\begin{aligned}
    Q x + A^T y + G^T z &= 0 \quad &\text{(Estacionariedad)} \\
    A x - b &= 0 \quad &\text{(Factibilidad Primal de Igualdad)} \\
    G x + s - h &= 0 \quad &\text{(Factibilidad Primal de Desigualdad)} \\
    S Z \mathbf{1} - \tau \mathbf{1} &= 0 \quad &\text{(Complementariedad perturbada)}
    \end{aligned}$$

    ---

    ### 4. Estabilidad Numérica y Factorización KKT
    Para resolver las direcciones de Newton $(\Delta x, \Delta y, \Delta z, \Delta s)$, condensamos el sistema a la forma simétrica aumentada indefinida:
    $$\begin{bmatrix} Q + G^T S^{-1} Z G & A^T \\ A & 0 \end{bmatrix} \begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix} = \begin{bmatrix} r_d - G^T S^{-1}(r_{sz} - Z r_p) \\ r_a \end{bmatrix}$$

    • **Factorización:** Se emplea la descomposición $LDL^T$ de Bunch-Kaufman con pivoteo simétrico.<br>
    • **Regularización Dinámica:** En caso de mal condicionamiento, se añade perturbación $-\epsilon I$ en los bloques duales.<br>
    • **Refinamiento Iterativo:** Corrige errores de redondeo de punto flotante para garantizar residuos KKT por debajo de $10^{-8}$.<br>
    • **Covarianza de Contracción Ledoit-Wolf:** Se estima $\Sigma_t$ sobre una ventana rodante de 252 días con shrinkage asintótico hacia la media de varianzas, garantizando condición estrictamente definida positiva sin sesgo de anticipación (*look-ahead bias*).
    """)


# --------------------------------------------------------------------------- #
# Barra Lateral: Exportación y Descarga de Resultados
# --------------------------------------------------------------------------- #
st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Exportar Datos")

csv_results = results_df.to_csv().encode("utf-8")
st.sidebar.download_button(
    label="📥 Descargar Rendimientos (CSV)",
    data=csv_results,
    file_name=f"mpo_rendimientos_L{L_param}_cost{cost_param}.csv",
    mime="text/csv"
)

csv_weights = weights_df.to_csv().encode("utf-8")
st.sidebar.download_button(
    label="📥 Descargar Pesos Históricos (CSV)",
    data=csv_weights,
    file_name=f"mpo_pesos_L{L_param}_cost{cost_param}.csv",
    mime="text/csv"
)

st.sidebar.caption("Antigravity Quantitative Analytics | PDIPM Engine")
