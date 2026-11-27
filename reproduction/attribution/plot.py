"""Plot concentration, lead time and within-channel drift from the attribution tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent / "report"


def plot():
    colors = {"ETTh1": "#12678a", "ETTm1": "#a45e0c", "Weather": "#725490"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), layout="constrained")
    concentration, bands, leads, drift = axes.ravel()
    for j, (name, color) in enumerate(colors.items()):
        folder = ROOT / name
        scores = pd.read_csv(folder / "concentration.csv")
        ci = pd.read_csv(folder / "intervals.csv").query("metric == 'total'").iloc[:4]
        x = np.arange(1, 5) + (j - 1) * 0.07
        concentration.plot(x, scores.delta, "o-", color=color, label=name)
        concentration.vlines(x, ci.low, ci.high, color=color, alpha=0.48)
        band_values = scores.loc[0, ["DC", "low", "mid", "high"]].astype(float)
        bands.bar(np.arange(1, 5) + (j - 1) * 0.23, band_values, width=0.22, color=color)
        steps = pd.read_csv(folder / "horizon.csv").query("q == 1")
        leads.plot(steps.lead, steps.delta, color=color, lw=1.5)
        by_drift = pd.read_csv(folder / "drift.csv").groupby("window_q").dc.mean()
        drift.plot(by_drift.index, by_drift, "o-", color=color)
    specs = [
        (
            concentration,
            "History spectral concentration",
            "Concentration quartile",
            "Total MSE difference",
        ),
        (
            bands,
            "Q1 error by forecast frequency",
            "Forecast error band",
            "Contribution to MSE difference",
        ),
        (leads, "Q1 error by lead time", "Forecast step", "MSE difference"),
        (
            drift,
            "Level error within each channel",
            "Historical mean drift quartile",
            "DC MSE difference",
        ),
    ]
    for ax, title, xlabel, ylabel in specs:
        ax.axhline(0, color="0.45", lw=0.8)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in (concentration, drift):
        ax.set_xticks(range(1, 5), ["Q1", "Q2", "Q3", "Q4"])
    bands.set_xticks(range(1, 5), ["DC", "Low", "Mid", "High"])
    leads.set_xticks([1, 24, 48, 72, 96])
    concentration.legend(frameon=False, fontsize=9)
    fig.suptitle("Paired test-window errors  ·  TimeFT minus PatchTST", fontsize=15)
    fig.savefig(ROOT / "attribution.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot()
