"""Alternative frozen readout, multi-horizon controls, and a locked 2022 evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from reproduction.mechanism.study import (
    EDGE, calibrate, choose_linear, coefficient_family, load_model,
    prepare_features, read_npz, rms, windows, write_json,
)


HORIZONS = (40, 80, 96, 120, 192)
VARIANTS = ("patch", "timeft")


def unpooled(model, x, projection, keep_full):
    encoded = []
    forecasts = []

    def observe(_module, _args, result):
        z = result[0] @ projection
        z = F.pad(z, (0, 0, 0, 20 - z.shape[1]))
        encoded.append(z.cpu().numpy())

    hook = model.encoder.register_forward_hook(observe)
    try:
        with torch.inference_mode():
            for i in range(0, len(x), 64):
                pred = model(torch.as_tensor(x[i:i + 64], device="cuda"), None, None, None)
                forecasts.append((pred if keep_full else pred.mean(1)).cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(encoded).reshape(x.shape[0], x.shape[2], -1), np.concatenate(forecasts)


def alternate_readout(args, data, records):
    basis, _ = np.linalg.qr(np.random.default_rng(2021).normal(size=(512, 48)))
    projection = torch.as_tensor(basis[:, :24], device="cuda", dtype=torch.float32)
    targets = {k: r["target"].reshape(-1, 2) for k, r in records.items()}
    high_cut = np.quantile(np.abs(records["train"]["recent"][..., 1]), .75, axis=0)
    record_22 = records["2022"]
    np.savez_compressed(args.output / "2022_truth.npz", time=record_22["time"], origin=record_22["origin"],
        true_mean=record_22["true_mean"], target=record_22["target"],
        high_shift=np.abs(record_22["recent"][..., 1]) >= high_cut)
    for variant in VARIANTS:
        model = load_model(args.checkpoints, "Weather", variant)
        features, forecasts = {}, {}
        for name, (series, split) in data.items():
            x, _ = windows(series, split)
            latent, forecasts[name] = unpooled(model, x, projection, name == "2022")
            scale = records[name]["scale"][..., None].astype(np.float32)
            features[name] = np.concatenate([latent * scale, scale], axis=-1).reshape(-1, 20 * 24 + 1)
            print(variant, "unpooled", name, len(x), flush=True)

        base2020 = np.load(args.primary / "Weather" / f"{variant}_test_pred.npy", mmap_mode="r")
        parity = rms(base2020.mean(1, dtype=np.float64) - forecasts["2020"])
        if parity > 1e-5:
            raise RuntimeError(f"Unexpected 2020 forecast mismatch in follow-up: {parity:.5g}")

        predictions, alpha = choose_linear(features["train"], targets["train"], features["val"], targets["val"],
            {name: features[name] for name in ("2020", "2021", "2022")})
        correction, tuning = calibrate({"val": {"forecast": forecasts["val"]}, "external": {"forecast": forecasts["2022"]}},
                                        {"val": records["val"], "external": record_22})
        orig2022 = forecasts["2022"]
        _, true2022 = windows(*data["2022"])
        error = orig2022.astype(np.float64) - true2022
        wave_mse = ((error - error.mean(1, keepdims=True)) ** 2).mean(1)
        out = {name: prediction.reshape(records[name]["target"].shape) for name, prediction in predictions.items()}
        np.savez_compressed(args.output / f"{variant}.npz", **out, **correction,
                            baseline=orig2022.mean(1, dtype=np.float64), wave_mse=wave_mse)
        np.save(args.output / f"{variant}_2022_pred.npy", orig2022, allow_pickle=False)
        write_json(args.output / f"{variant}_settings.json", {"unpooled_alpha": alpha, "calibration": tuning,
                                                                "weather_2020_parity_rms": parity})
        del model, features, forecasts


def natural_horizons(data, output):
    features, targets = {}, {}
    ends = {"train": 36887, "val": 42157, "2020": 52696}
    for name, (series, split) in data.items():
        origins = series[split]
        longest = max(HORIZONS)
        origins = origins[origins + longest <= ends.get(name, len(series["values"]))]
        values = series["values"]
        longest_target = values[origins[:, None] + np.arange(longest)]
        full_span = series["time"][origins + longest - 1] - series["time"][origins - 320]
        keep = np.isfinite(longest_target).all(axis=(1, 2)) & (full_span == (320 + longest - 1) * series["minutes"])
        origins = origins[keep]
        steps = longest_target[keep]
        x = values[origins[:, None] + np.arange(-320, 0)]
        mean = x.mean(1, dtype=np.float64)
        features[name] = x - mean[:, None, :]
        targets[name] = np.stack([steps[:, :h].mean(1, dtype=np.float64) - mean for h in HORIZONS], axis=-1)

    scores, estimates = [], {}
    for name, bins in (("full", None), ("edge_removed", EDGE), ("lower_removed", EDGE - 1), ("upper_removed", EDGE + 1)):
        arrays = {key: (coefficient_family(x, bins, "remove") if bins is not None else x).transpose(0, 2, 1).reshape(-1, 320)
                  for key, x in features.items()}
        forecasts, alpha = choose_linear(arrays["train"], targets["train"].reshape(-1, len(HORIZONS)),
            arrays["val"], targets["val"].reshape(-1, len(HORIZONS)), {key: arrays[key] for key in ("2020", "2021", "2022")})
        for year, forecast in forecasts.items():
            estimate = forecast.reshape(targets[year].shape)
            estimates[f"{year}_{name}"] = estimate
            loss = ((estimate - targets[year]) ** 2).mean(axis=(0, 1))
            scores.extend({"year": year, "input": name, "horizon": horizon, "mse": float(mse),
                           "alpha": penalty, "windows": len(estimate)}
                          for horizon, mse, penalty in zip(HORIZONS, loss, alpha))
        print("Natural horizon controls", name, flush=True)
    np.savez_compressed(output / "horizon_predictions.npz", **estimates,
                        **{f"{y}_truth": targets[y] for y in ("2020", "2021", "2022")})
    write_json(output / "natural_horizons.json", scores)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(1)
    weather = read_npz(args.inputs / "Weather.npz")
    weather21 = read_npz(args.inputs / "Weather2021.npz")
    weather22 = read_npz(args.external)
    data = {"train": (weather, "train"), "val": (weather, "val"), "2020": (weather, "test"),
            "2021": (weather21, "test"), "2022": (weather22, "test")}
    records = {name: prepare_features(*entry)[2] for name, entry in data.items()}
    alternate_readout(args, data, records)
    natural_horizons(data, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "checkpoints", "primary", "external", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    main(parser.parse_args())
