"""Prepare the locked second weather holdout using the original scaler."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reproduction.mechanism import prepare


ARCHIVES_2022 = {
    "2022a": "bc329f337ff696ababae0c5ebc279039b2b5f9cea07f5b60e96659ff7dd81656",
    "2022b": "980c6ef21c7b4bb8c9d49d50c8dc81a7c4c6b86036bab169b392791a1870fd1a",
}


def main(project, output):
    output.mkdir(parents=True, exist_ok=True)
    orig = json.loads((project / "results/table5/formal/Weather/data.json").read_text(encoding="utf-8"))
    archive_dir = project / "dataset/mechanism/weather-mpi"
    prepare.ARCHIVES.update(ARCHIVES_2022)
    frame = pd.concat([prepare.read_roof(archive_dir, period) for period in ARCHIVES_2022], ignore_index=True)
    frame = frame.loc[frame["date"] > pd.Timestamp("2022-01-01 00:00")].sort_values("date")
    duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates()
    if frame["date"].duplicated().any() or frame["date"].max() > pd.Timestamp("2023-01-01 00:00"):
        raise ValueError("Invalid or duplicate holdout timestamp")
    channels = orig["channels"]
    raw = frame[channels].to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(raw) | (raw <= -9990)
    values = (np.where(invalid, np.nan, raw) - orig["scale_mean"]) / orig["scale_std"]
    t = frame["date"].to_numpy(dtype="datetime64[m]").astype(np.int64)
    origins = prepare.valid_origins(values, t, 320, len(t), 10, 6)
    if len(origins) < 5000:
        raise ValueError("Insufficient valid 2022 Weather windows")
    result = prepare.write_dataset(output, "Weather2022", values, t, {"test": origins}, channels, 10)
    evidence = {"first": str(frame["date"].min()), "last": str(frame["date"].max()), "rows": len(frame),
                "bad_rows": int(invalid.any(axis=1).sum()), "exact_duplicates_removed": duplicate_rows,
                "missing_10min_rows": int((t[-1] - t[0]) // 10 + 1 - len(t)), "windows": result,
                "sources": [{"url": f"{prepare.SOURCE}mpi_roof_{k}.zip", "sha256": digest, "license": "CC BY 4.0"}
                            for k, digest in ARCHIVES_2022.items()]}
    (output / "input2022.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--project", default=Path("."), type=Path)
    args = parser.parse_args()
    main(args.project, args.output)
