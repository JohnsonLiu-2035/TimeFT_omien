"""Build the mechanism study inputs using the saved Table 5 scalers and splits."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


SOURCE = "https://www.bgc-jena.mpg.de/wetter/data/"
ARCHIVES = {
    "2020a": "6a0be0552a2a26a04c3b41a5281d4a55ae251d4c65ce630971a8662f43b31f9f",
    "2021a": "3785931544fef3e66166e0b3cb1ec4496ee5e92042e7de0d1a094fac64311668",
    "2021b": "772869a7204f2bdf95154334f6314c3cd8a32c3c4df44d95ffa4ab904a5b058c",
}
SPECS = {
    "Weather": ("weather/weather.csv", 10, (36887, 42157, 52696), 6),
    "ETTh1": ("ETT-small/ETTh1.csv", 60, (8640, 11520, 14400), 4),
    "ETTm1": ("ETT-small/ETTm1.csv", 15, (34560, 46080, 57600), 6),
}


def checksum(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_roof(root, period):
    archive = root / f"mpi_roof_{period}.zip"
    actual = checksum(archive)
    if actual != ARCHIVES[period]:
        raise ValueError(f"Unexpected source SHA-256 for {archive}")
    with ZipFile(archive) as zipped, zipped.open(f"mpi_roof_{period}.csv") as stream:
        frame = pd.read_csv(stream, encoding="latin-1")
    frame["date"] = pd.to_datetime(frame.pop("Date Time"), format="%d.%m.%Y %H:%M:%S")
    frame = frame.rename(columns=lambda title: title.replace("\u00b2", "\ufffd").replace("\u00b5", "\ufffd"))
    return frame.rename(columns={"CO2 (ppm)": "OT"})


def valid_origins(values, time, start, end, minutes, step):
    # Both the history and target must be complete and span the intended duration.
    candidates = np.arange(start, end - 96 + 1, step)
    impossible = ~np.isfinite(values).all(axis=1)
    bad_rows = np.r_[0, np.cumsum(impossible)]
    discontinuities = np.r_[0, np.cumsum(np.diff(time) != minutes)]
    left, right = candidates - 320, candidates + 96 - 1
    matches = (bad_rows[right + 1] == bad_rows[left]) & (discontinuities[right] == discontinuities[left])
    return candidates[matches]


def write_dataset(output, name, values, times, splits, channel_names, minutes):
    np.savez_compressed(output / f"{name}.npz", values=values.astype(np.float32), time=times,
                        channels=np.asarray(channel_names), minutes=np.int64(minutes), **splits)
    return {flag: len(indices) for flag, indices in splits.items()}


def main(project, output):
    output.mkdir(parents=True, exist_ok=False)
    citations = {"benchmark": {}, "weather_archives": [], "external": {}}
    for dataset, (filename, minutes, ends, step) in SPECS.items():
        formal = project / "results" / "table5" / "formal" / dataset
        metadata = json.loads((formal / "data.json").read_text(encoding="utf-8"))
        csv_file = project / "dataset" / filename
        assert checksum(csv_file) == metadata["sha256"], csv_file
        heads = {}
        for variant in ("patch", "timeft"):
            heads[variant] = {file: checksum(formal / variant / file) for file in ("checkpoint.pth", "config.json")}
        cols = metadata["channels"]
        scale_mean = np.asarray(metadata["scale_mean"])
        scale_std = np.asarray(metadata["scale_std"])
        frame = pd.read_csv(csv_file)
        numeric = frame[cols].to_numpy(dtype=np.float64)
        assert np.isfinite(numeric).all()
        standardized = (numeric - scale_mean) / scale_std
        timestamps = pd.to_datetime(frame["date"]).to_numpy(dtype="datetime64[m]").astype(np.int64)
        splits = {name: valid_origins(standardized, timestamps, start, end, minutes, step)
                  for name, start, end in zip(("train", "val", "test"), (320, *ends[:2]), ends)}
        assert np.all(splits["test"] >= metadata["raw_test_target_start"])
        index = splits["test"] - metadata["raw_test_target_start"]
        # Keep a numerical fingerprint of predictions from the original GPU run.
        prior = {variant: np.load(formal / variant / "pred.npy", mmap_mode="r")[index[:32]]
                 for variant in ("patch", "timeft")}
        np.savez_compressed(output / f"{dataset}-reference.npz", **prior)
        citations["benchmark"][dataset] = {
            "path": filename, "dataset_sha256": metadata["sha256"], "model_artifacts": heads,
            "step": step, "windows": write_dataset(output, dataset, standardized, timestamps, splits, cols, minutes),
        }
        if dataset != "Weather":
            continue

        roof_dir = project / "dataset" / "mechanism" / "weather-mpi"
        anchor = read_roof(roof_dir, "2020a").drop_duplicates(subset="date", keep=False)
        matched = frame["date"].pipe(pd.to_datetime).map(pd.Series(np.arange(len(anchor)), index=anchor["date"]))
        check_rows = np.flatnonzero(matched.notna())
        original_rows = anchor.iloc[matched.iloc[check_rows].astype(int).to_numpy()][cols].to_numpy()
        equal_fraction = np.isclose(numeric[check_rows], original_rows, rtol=0, atol=1e-5).mean(axis=0)
        if len(check_rows) < 20000 or np.min(equal_fraction) < 0.99:
            raise ValueError(f"Weather station/schema match failed: {dict(zip(cols, equal_fraction))}")

        external = pd.concat([read_roof(roof_dir, period) for period in ("2021a", "2021b")], ignore_index=True)
        # Files are labelled by acquisition period. Enforce a disjoint timestamp interval.
        external = external.loc[external["date"] > pd.to_datetime(frame["date"]).max()].sort_values("date")
        if external["date"].duplicated().any():
            raise ValueError("Duplicate external timestamps")
        raw = external[cols].to_numpy(dtype=np.float64)
        corrupt = (~np.isfinite(raw)) | (raw <= -9990)
        clean = np.where(corrupt, np.nan, raw)
        ext_values = (clean - scale_mean) / scale_std
        ext_times = external["date"].to_numpy(dtype="datetime64[m]").astype(np.int64)
        ext_split = {"test": valid_origins(ext_values, ext_times, 320, len(ext_times), minutes, step)}
        if len(ext_split["test"]) < 5000:
            raise ValueError("Fewer than 5,000 valid external windows")
        citations["external"] = {
            "matched_benchmark_rows": len(check_rows), "minimum_channel_match": float(equal_fraction.min()),
            "OT_alias": "CO2 (ppm)", "scale": "Table 5 Weather 2020 training scaler",
            "first": str(external["date"].min()), "last": str(external["date"].max()),
            "rows": len(raw), "bad_rows": int(corrupt.any(axis=1).sum()),
            "examined": (len(raw) - 96 - 320) // step + 1,
            "windows": write_dataset(output, "Weather2021", ext_values, ext_times, ext_split, cols, minutes),
        }
        citations["weather_archives"] = [
            {"url": f"{SOURCE}mpi_roof_{period}.zip", "sha256": value, "license": "CC BY 4.0"}
            for period, value in ARCHIVES.items()
        ]
    (output / "inputs.json").write_text(json.dumps(citations, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(citations, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--project", default=Path("."), type=Path)
    args = parser.parse_args()
    main(args.project, args.output)
