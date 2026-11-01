"""
figuras_paper.py
================
Figuras de resultados del informe, a partir del motor de backtest común.

    python figuras_paper.py                 # guarda las figuras en figures/
    python figuras_paper.py --out mi_carpeta

Requiere haber corrido antes dataset.py y sensitivity.py (lee sensibilidad_resultados.csv).
Genera:
  * fig_acumulado.png       retorno neto acumulado (L = 0.01, 0.05, 1) vs. 1/N y SPY
  * fig_sensibilidad_L.png  turnover diario y volatilidad anualizada en función de L
  * fig_sectores.png        peso por sector del portafolio a lo largo del tiempo (L = 0.05)
e imprime estadísticas de composición del portafolio con L = 0.05.
La figura de la comparación de solvers la genera benchmark.py (benchmark_3way_comparison.png).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

import config  # noqa: E402
from backtest_engine import (BacktestResult, load_market_data, make_solver,  # noqa: E402
                             reference_returns, run_backtest)

L_CURVAS = [(0.01, "tab:blue"), (0.05, "tab:green"), (1.0, "tab:red")]
L_COMPOSICION = 0.05
SECTORES_ES = {
    "Information Technology": "Tecnología", "Health Care": "Salud", "Financials": "Finanzas",
    "Consumer Discretionary": "Consumo discrecional", "Consumer Staples": "Consumo básico",
    "Industrials": "Industria", "Utilities": "Servicios públicos", "Real Estate": "Inmobiliario",
    "Materials": "Materiales", "Energy": "Energía", "Communication Services": "Comunicaciones",
}


def figura_acumulado(data, resultados: Dict[float, BacktestResult], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for L, color in L_CURVAS:
        ax.plot(data.dates, np.cumprod(1.0 + resultados[L].net_returns) - 1.0, color=color, lw=1.4,
                label=f"Mínima varianza, L = {L:g}")
    for (name, s), (color, ls) in zip(reference_returns(data).items(), [("dimgray", "--"), ("black", ":")]):
        label = "1/N (sin costos)" if name.startswith("1/N") else "SPY (sin costos)"
        ax.plot(data.dates, np.cumprod(1.0 + s) - 1.0, color=color, ls=ls, lw=1.2, label=label)
    ax.set_ylabel("Retorno acumulado")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "fig_acumulado.png", dpi=200)


def figura_sensibilidad(out: Path) -> None:
    df = pd.read_csv("sensibilidad_resultados.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(df["L"], df["Turnover diario (%)"], "o-", color="tab:blue")
    axes[0].plot(df["L"], 100 * df["L"], ":", color="gray", label="Límite L")
    axes[0].set_ylim(0, 4.5)
    axes[0].set_ylabel("Turnover diario promedio (%)")
    axes[0].legend(fontsize=9)
    axes[1].plot(df["L"], df["Volatilidad anual (%)"], "o-", color="tab:red")
    axes[1].set_ylim(9.5, 10.2)
    axes[1].set_ylabel("Volatilidad anualizada (%)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(df["L"])
        ax.set_xticklabels([f"{v:g}" for v in df["L"]])
        ax.set_xlabel("Presupuesto de turnover L")
        ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out / "fig_sensibilidad_L.png", dpi=200)


def composicion(res: BacktestResult, out: Path) -> None:
    W = pd.DataFrame(res.weights, index=res.dates, columns=res.tickers)
    top10 = W.apply(lambda r: r.nlargest(10).sum(), axis=1)
    print(f"Composición con L = {L_COMPOSICION}:")
    print(f"  activos con peso > 0.1%: media {(W > 1e-3).sum(axis=1).mean():.1f}")
    print(f"  número efectivo 1/sum(w^2): media {(1.0 / (W ** 2).sum(axis=1)).mean():.1f}")
    print(f"  peso de los 10 mayores: media {100 * top10.mean():.1f}%")

    uni = pd.read_csv(config.UNIVERSE_FILE).set_index("Symbol")
    sector = uni.reindex(W.columns)["GICS Sector"].fillna("Otro")
    S = W.T.groupby(sector).sum().T
    proporcion = sector.value_counts(normalize=True)
    orden = S.mean().sort_values(ascending=False)
    print("  peso promedio por sector (portafolio vs. proporción de activos en el universo):")
    for s, v in orden.items():
        print(f"    {s:28s} {100 * v:5.1f}%   {100 * proporcion.get(s, 0.0):5.1f}%")
    print("  mayores pesos promedio:")
    for t, v in W.mean().nlargest(10).items():
        print(f"    {t:6s} {sector[t]:28s} {100 * v:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colores = plt.cm.tab20(np.linspace(0, 1, len(orden)))
    ax.stackplot(S.index, [100 * S[s] for s in orden.index],
                 labels=[SECTORES_ES.get(s, s) for s in orden.index], colors=colores, alpha=0.9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Peso en el portafolio (%)")
    ax.margins(x=0)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "fig_sectores.png", dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures", help="carpeta de salida de las figuras")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11})

    data = load_market_data()
    solver = make_solver()
    resultados = {L: run_backtest(data, L, cost_rate=config.COST_RATE, solver=solver) for L, _ in L_CURVAS}

    figura_acumulado(data, resultados, out)
    figura_sensibilidad(out)
    composicion(resultados[L_COMPOSICION], out)
    print(f"Figuras guardadas en '{out}/'.")


if __name__ == "__main__":
    main()
