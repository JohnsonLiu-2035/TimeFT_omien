"""Plot the measured training and validation losses at every completed epoch."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot(folder):
    data = pd.read_csv(folder / "epochs.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained", sharex="row")
    for row, name in enumerate(("Weather", "ETTh1")):
        for col, (metric, title) in enumerate(
            (("train_mse", "Training"), ("val_batch_mean_mse", "Validation"))
        ):
            ax = axes[row, col]
            for (model, schedule), group in data[data.dataset == name].groupby(["model", "schedule"]):
                color = {"patch": "#216786", "timeft": "#b25334"}[model]
                style = {"cosine": "-", "type1": "--"}[schedule]
                ax.plot(group.epoch, group[metric], color=color, ls=style, label=f"{model} / {schedule}")
                if col == 1:
                    best = group.loc[group[metric].idxmin()]
                    ax.scatter(
                        best.epoch,
                        best[metric],
                        color=color,
                        s=32,
                        marker="o" if schedule == "cosine" else "x",
                    )
            ax.set(xlabel="Epoch", ylabel="Batch-mean MSE", title=f"{name}   ·   {title}")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(folder / "learning_curves.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    plot(Path("reproduction/lr_schedule/report"))
