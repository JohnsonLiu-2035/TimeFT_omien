"""Compare the four cosine runs with the archived type1 baselines on paired test origins."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reproduction.attribution.analyze import concentration, quartiles


OUT = Path("reproduction/lr_schedule/report")
DATASETS = ("ETTh1", "Weather")
SCHEDULES = ("type1", "cosine")
MODELS = ("patch", "timeft")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_runs(old, new):
    for name in ("target_row.npy", "target_time.npy", "true.npy"):
        np.testing.assert_array_equal(
            np.load(old / name, mmap_mode="r"), np.load(new / name, mmap_mode="r")
        )
    assert read_json(old / "data.json") == read_json(new / "data.json")
    for model in MODELS:
        a, b = (read_json(root / model / "config.json") for root in (old, new))
        assert a["lradj"] == "type1" and b["lradj"] == "cosine"
        excluded = {"lradj", "data_dir", "root_path", "output", "gpu"}
        assert {k: v for k, v in a.items() if k not in excluded} == {
            k: v for k, v in b.items() if k not in excluded
        }, (old, model)
        assert a["seed"] == 2021 and not b["smoke"]
        assert (b["learning_rate"], b["train_epochs"], b["patience"]) == (5e-5, 70, 5)


def window_metrics(pred, truth):
    assert pred.shape == truth.shape
    values = {}
    for i in range(0, len(truth), 128):
        y = truth[i : i + 128].astype(float)
        p = pred[i : i + 128].astype(float)
        yc, pc = y - y.mean(1, keepdims=True), p - p.mean(1, keepdims=True)
        var = (pc**2).mean(1)
        denom = np.sqrt(var * (yc**2).mean(1))
        cov = (pc * yc).mean(1)
        corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 1e-12)
        metrics = {
            "mse": ((p - y) ** 2).mean((1, 2)),
            "dc": ((p - y).mean(1) ** 2).mean(1),
            "shape": ((pc - yc) ** 2).mean((1, 2)),
            "variance": var.mean(1),
            "roughness": (np.diff(p, axis=1) ** 2).mean((1, 2)),
            "correlation": corr,
        }
        for key, value in metrics.items():
            values.setdefault(key, []).append(value)
    values = {key: np.concatenate(parts) for key, parts in values.items()}
    np.testing.assert_allclose(values["mse"], values["dc"] + values["shape"], atol=1e-12)
    return values


def paired_interval(values, mask, block=416):
    """As in attribution, sample contiguous circular blocks of paired forecast origins."""
    n = len(values)
    samples = np.column_stack([values * mask, mask])
    partial = np.vstack([np.zeros(2), np.vstack([samples, samples[:block]]).cumsum(axis=0)])
    totals = partial[block : block + n] - partial[:n]
    starts = np.random.default_rng(2021).integers(n, size=(2000, int(np.ceil(n / block))))
    sums = totals[starts].sum(1)
    draws = sums[sums[:, 1] > 0]
    return np.quantile(draws[:, 0] / draws[:, 1], [0.025, 0.975])


def analyze(dataset, levels, contrasts, dynamics, epochs):
    original = Path("results/table5/formal") / dataset
    baseline = Path("results/attribution/clean/formal/Weather") if dataset == "Weather" else original
    roots = {"type1": baseline, "cosine": Path("results/lr_schedule/formal") / dataset}
    check_runs(baseline, roots["cosine"])
    truth = np.load(baseline / "true.npy", mmap_mode="r")
    truth_var = np.concatenate(
        [chunk.astype(float).var(1).mean(1) for chunk in np.array_split(truth, 100)]
    )
    truth_diff = np.concatenate(
        [
            (np.diff(chunk.astype(float), axis=1) ** 2).mean((1, 2))
            for chunk in np.array_split(truth, 100)
        ]
    )
    history = np.load(roots["cosine"] / "history.npy", mmap_mode="r")
    groups = quartiles(
        np.concatenate([concentration(history[i : i + 128]) for i in range(0, len(truth), 128)])
    )
    results = {}
    for schedule, root in roots.items():
        for model in MODELS:
            trial = root / model
            config, summary = (read_json(trial / name) for name in ("config.json", "summary.json"))
            values = window_metrics(np.load(trial / "pred.npy", mmap_mode="r"), truth)
            results[schedule, model] = values
            np.testing.assert_allclose(values["mse"].mean(), summary["mse"], rtol=1e-9)
            frame = pd.read_csv(trial / "epochs.csv")
            best = frame.loc[frame["val_batch_mean_mse"].idxmin()]
            stop = frame.iloc[-1]
            assert summary["best_epoch"] == int(best.epoch) and summary["epochs"] == int(stop.epoch)
            assert len(frame) == 70 or int(stop.epoch - best.epoch) == 5
            epoch = frame.epoch.to_numpy()
            rate = (
                5e-5 * 0.5 ** np.maximum(0, epoch - 2)
                if schedule == "type1"
                else 5e-5 * (1 + np.cos(np.pi * (epoch - 1) / 70)) / 2
            )
            np.testing.assert_allclose(frame.lr, rate, atol=1e-18)
            epochs.append(frame.assign(dataset=dataset, schedule=schedule, model=model))
            dynamics.append(
                dict(
                    dataset=dataset,
                    schedule=schedule,
                    model=model,
                    gpu=config["gpu"],
                    best=int(best.epoch),
                    stop=int(stop.epoch),
                    lr_best=best.lr,
                    initial_train=frame.train_mse.iloc[0],
                    train_best=best.train_mse,
                    final_train=stop.train_mse,
                    initial_val=frame.val_batch_mean_mse.iloc[0],
                    best_val=best.val_batch_mean_mse,
                    final_val=stop.val_batch_mean_mse,
                )
            )
    common = np.logical_and.reduce([np.isfinite(v["correlation"]) for v in results.values()])
    for subset, mask in [("all", np.ones(len(truth), dtype=bool)), ("Q1", groups == 1)]:
        for (schedule, model), vals in results.items():
            row = {k: v[mask].mean() for k, v in vals.items() if k != "correlation"}
            levels.append(
                dict(
                    dataset=dataset,
                    schedule=schedule,
                    model=model,
                    subset=subset,
                    truth_variance=truth_var[mask].mean(),
                    truth_roughness=truth_diff[mask].mean(),
                    windows=int(mask.sum()),
                    correlation=vals["correlation"][mask][common[mask]].mean(),
                    correlation_coverage=common[mask].mean(),
                    **row,
                )
            )
        for key in ("mse", "dc", "shape", "variance", "roughness"):
            gaps = {s: results[s, "timeft"][key] - results[s, "patch"][key] for s in SCHEDULES}
            changes = {
                m + "_schedule_change": results["cosine", m][key] - results["type1", m][key]
                for m in MODELS
            }
            for label, vector in [
                *gaps.items(),
                ("cosine_minus_type1", gaps["cosine"] - gaps["type1"]),
                *changes.items(),
            ]:
                low, high = paired_interval(vector, mask)
                contrasts.append(
                    dict(
                        dataset=dataset,
                        subset=subset,
                        comparison=label,
                        metric=key,
                        delta=vector[mask].mean(),
                        ci_low=low,
                        ci_high=high,
                    )
                )


if __name__ == "__main__":
    levels, contrasts, dynamics, epochs = [], [], [], []
    for dataset in DATASETS:
        analyze(dataset, levels, contrasts, dynamics, epochs)
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("metrics", levels), ("contrasts", contrasts), ("training", dynamics)):
        pd.DataFrame(rows).to_csv(OUT / (filename + ".csv"), index=False)
    pd.concat(epochs, ignore_index=True).to_csv(OUT / "epochs.csv", index=False)
