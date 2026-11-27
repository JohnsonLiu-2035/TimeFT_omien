"""Compare original and cleaned-training forecasts in the same units on paired origins."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reproduction.attribution.analyze import concentration, intervals, quartiles


def compare(original, cleaned, destination):
    def metadata(folder):
        return json.loads((folder / "data.json").read_text())

    first, second = metadata(original), metadata(cleaned)
    assert first["channels"] == second["channels"]
    for variant in ("patch", "timeft"):
        configs = [
            json.loads((folder / variant / "config.json").read_text()) for folder in (original, cleaned)
        ]
        for field in (
            "seed",
            "windows",
            "learning_rate",
            "lradj",
            "train_epochs",
            "patience",
            "batch_size",
            "encoder_initial_sha256",
            "parameters",
            "seq_len",
            "pred_len",
            "matmul_precision",
        ):
            assert configs[0][field] == configs[1][field], (variant, field)
    for name in ("target_row.npy", "target_time.npy"):
        np.testing.assert_array_equal(np.load(original / name), np.load(cleaned / name))
    old_std, new_std = (np.array(meta["scale_std"]) for meta in (first, second))
    raw_truth = np.load(original / "true.npy", mmap_mode="r")
    clean_truth = np.load(cleaned / "true.npy", mmap_mode="r")
    assert raw_truth.shape == clean_truth.shape == (10444, 96, 21)
    for i in range(0, len(raw_truth), 128):
        sl = slice(i, i + 128)
        np.testing.assert_allclose(
            raw_truth[sl] * old_std + first["scale_mean"],
            clean_truth[sl] * new_std + second["scale_mean"],
            atol=2e-4,
            rtol=2e-6,
        )
    history = np.load(original / "history.npy", mmap_mode="r")
    drift = np.abs(history[:, 160:].mean(1, dtype=float) - history[:, :160].mean(1, dtype=float))
    ranks = np.column_stack([quartiles(series) for series in drift.T])
    groups = quartiles(np.concatenate([concentration(history[i:i+128]) for i in range(0,len(history),128)]))
    shape_rows = []
    rows, channels, drift_rows = [], [], []
    for label, folder, std, y in (
        ("original", original, old_std, raw_truth),
        ("cleaned", cleaned, new_std, clean_truth),
    ):
        by_variant = {}
        for variant in ("patch", "timeft"):
            p = np.load(folder / variant / "pred.npy", mmap_mode="r")
            assert p.shape == y.shape
            total, level = [], []
            for i in range(0, len(y), 128):
                sl = slice(i, i + 128)
                residual = p[sl].astype(float) - y[sl]
                total.append((residual**2).mean(1))
                level.append(residual.mean(1) ** 2)
            by_variant[variant] = np.stack([np.concatenate(total), np.concatenate(level)])
            expected = json.loads((folder / variant / "summary.json").read_text())["mse"]
            np.testing.assert_allclose(by_variant[variant][0].mean(), expected, rtol=1e-9)
        for unit, scale in (("original", old_std), ("cleaned", new_std)):
            weighted = {v: a * (std / scale) ** 2 for v, a in by_variant.items()}
            difference = weighted["timeft"] - weighted["patch"]
            for q in range(1, 5):
                drift_rows.append(
                    dict(training=label, units=unit, window_q=q, dc=difference[1][ranks == q].mean())
                )
            window = difference.mean(2)
            shape = window[0] - window[1]
            bounds = intervals(shape, groups)
            means = [shape[groups == q].mean() for q in range(1,5)]
            for j, value in enumerate([*means, means[0]-means[3]]):
                shape_rows.append(dict(training=label, units=unit, group=("Q1","Q2","Q3","Q4","Q1-Q4")[j],
                                       shape=value, low=bounds[0,j], high=bounds[1,j]))
            n = window.shape[1]
            block = 416
            extended = np.hstack([window, window[:, :block]])
            partial = np.column_stack([np.zeros(2), extended.cumsum(1)])
            means = (partial[:, block : block + n] - partial[:, :n]) / block
            starts = np.random.default_rng(2021).integers(n, size=(2000, int(np.ceil(n / block))))
            bounds = np.quantile(means[:, starts].mean(2), [0.025, 0.975], axis=1)
            for k, metric in enumerate(["total", "dc"]):
                rows.append(
                    dict(
                        training=label,
                        units=unit,
                        metric=metric,
                        patch=weighted["patch"][k].mean(),
                        timeft=weighted["timeft"][k].mean(),
                        delta=window[k].mean(),
                        low=bounds[0, k],
                        high=bounds[1, k],
                    )
                )
                for ch, val in zip(first["channels"], difference[k].mean(0)):
                    channels.append(
                        dict(training=label, units=unit, metric=metric, channel=ch, delta=val)
                    )
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / "control.csv", index=False)
    pd.DataFrame(channels).to_csv(destination / "control_channels.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(destination / "control_drift.csv", index=False)
    pd.DataFrame(shape_rows).to_csv(destination / "control_shape.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("cleaned", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    compare(args.original, args.cleaned, args.destination)
