"""
sensitivity.py
==============
Barrido del presupuesto de turnover L sobre el motor de backtest común.

    python sensitivity.py                 # guarda PNG y CSV
    python sensitivity.py --show          # además abre la ventana de matplotlib
    python sensitivity.py --cost-bps 20 --init equal
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

import config
from backtest_engine import (compute_metrics, load_market_data,
                             make_solver, reference_returns, run_backtest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost-bps", type=float, default=config.COST_RATE * 1e4)
    ap.add_argument("--init", choices=["minvar", "equal"], default="minvar")
    ap.add_argument("--show", action="store_true", help="mostrar la figura en pantalla")
    args = ap.parse_args()

    data = load_market_data()
    T, N = data.returns.shape
    cost = args.cost_bps * 1e-4
    solver = make_solver()
    print(f"--- Sensibilidad: {len(config.L_GRID)} "
          f"niveles de L | T={T} días, N={N}, costo={args.cost_bps:.1f} bps ---")

    rows: List[Dict[str, float]] = []
    cum: Dict[str, NDArray[np.float64]] = {}
    for L in config.L_GRID:
        res = run_backtest(data, L, cost_rate=cost, init=args.init, solver=solver)
        m = compute_metrics(res.net_returns)
        solved = res.iterations[res.iterations > 0]
        rows.append({
            "L": L,
            "Retorno neto anual (%)": 100 * m["ann_return"],
            "Volatilidad anual (%)": 100 * m["ann_vol"],
            "Sharpe neto": m["sharpe"],
            "Sharpe SE": m["sharpe_se"],
            "Max drawdown (%)": 100 * m["max_drawdown"],
            "Turnover diario (%)": 100 * res.turnover.mean(),
            "Iter. solver": float(solved.mean()) if solved.size else float("nan"),
            "Fallos": float(res.failed.sum()),
        })
        cum[f"L = {L}"] = np.cumprod(1.0 + res.net_returns) - 1.0

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("TABLA DE SENSIBILIDAD (retornos netos de costos)")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    refs = reference_returns(data)
    ref_rows = {name: compute_metrics(s) for name, s in refs.items()}
    print("\nReferencias (SIN costos de transacción; el portafolio de arriba es neto):")
    ref_df = pd.DataFrame(ref_rows).T[["ann_return", "ann_vol", "sharpe", "sharpe_se", "max_drawdown"]]
    print(ref_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nNota: con ~2 años de datos el Sharpe SE suele ser ~0.7; diferencias menores "
          "entre filas no son estadísticamente distinguibles.")

    df.to_csv("sensibilidad_resultados.csv", index=False)

    # ---------------------------- Gráficas ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for label, c in cum.items():
        axes[0].plot(data.dates, c, label=label)
    for name, s in refs.items():
        axes[0].plot(data.dates, np.cumprod(1.0 + s) - 1.0, "--", color="gray", alpha=0.7, label=name)
    axes[0].set_title("Retorno neto acumulado por nivel de L")
    axes[0].set_xlabel("Fecha")
    axes[0].set_ylabel("Retorno acumulado")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)
    fig.autofmt_xdate()

    axes[1].errorbar(df["Turnover diario (%)"], df["Sharpe neto"], yerr=df["Sharpe SE"],
                     fmt="o", color="red", ecolor="lightcoral", capsize=3)
    for _, row in df.iterrows():
        axes[1].annotate(f"L={row['L']}", (row["Turnover diario (%)"], row["Sharpe neto"]),
                         textcoords="offset points", xytext=(5, 5))
    axes[1].set_title("Sharpe neto vs. turnover diario (barras = ±1 error estándar)")
    axes[1].set_xlabel("Turnover promedio diario (%)")
    axes[1].set_ylabel("Sharpe neto")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("analisis_sensibilidad.png", dpi=150)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()