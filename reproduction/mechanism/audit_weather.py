"""Audit the impact of the benchmark's missing-value sentinels on evaluation weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main(project, destination):
    formal = project / "results/table5/formal/Weather"
    metadata = json.loads((formal / "data.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(project / "dataset/weather/weather.csv")[metadata["channels"]]
    count = {split: int((raw.iloc[left:right] <= -9990).any(axis=1).sum())
             for split, left, right in (("train", 0, 36887), ("val", 36887, 42157), ("test", 42157, 52696))}
    train = raw.iloc[:36887]
    clean = train.where(train > -9990)
    clean_std = clean.std(ddof=0).to_numpy()
    original_std = np.asarray(metadata["scale_std"])
    ratio = original_std / clean_std
    shifts = {channel: {"missing_train": int(clean[channel].isna().sum()), "original_std": original_std[i],
                        "clean_std": clean_std[i], "original_to_clean": ratio[i]}
              for i, channel in enumerate(metadata["channels"]) if clean[channel].isna().any()}
    origin = np.load(formal / "target_row.npy", mmap_mode="r")
    bad = np.r_[0, (raw <= -9990).any(axis=1).to_numpy().cumsum()]
    test_with_sentinel = int(np.count_nonzero(bad[origin + 96] != bad[origin - 320]))
    truth = np.load(formal / "true.npy", mmap_mode="r")
    scores = {}
    for variant in ("patch", "timeft"):
        pred = np.load(formal / variant / "pred.npy", mmap_mode="r")
        error = pred.astype(np.float64) - truth
        total = (error ** 2).mean(axis=(0, 1))
        level = (error.mean(axis=1) ** 2).mean(axis=0)
        scores[variant] = {"original_total_mse": float(total.mean()), "original_mean_mse": float(level.mean()),
                           "clean_scale_total_mse": float((total * ratio ** 2).mean()),
                           "clean_scale_mean_mse": float((level * ratio ** 2).mean())}
    report = {"rows_with_sentinel": count, "test_windows_with_sentinel": test_with_sentinel,
              "std_inflation": shifts, "scores": scores}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--project", type=Path, default=Path("."))
    args = parser.parse_args()
    main(args.project, args.destination)
