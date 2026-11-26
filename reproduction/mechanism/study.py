"""Run the pre-specified phase interventions and frozen readouts on the saved models."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from reproduction.table5.model import PairedPatchTST


LENGTH, HORIZON, BAND = 320, 96, 8
EDGE = np.arange(BAND, LENGTH // 2, BAND)
ALPHAS = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10.])
VARIANTS = ("patch", "timeft")
DATASETS = ("Weather", "ETTh1", "ETTm1")


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def windows(series, split):
    origin = series[split]
    x = series["values"][origin[:, None] + np.arange(-LENGTH, 0)]
    y = series["values"][origin[:, None] + np.arange(HORIZON)]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite values passed the input window filter")
    return x, y


def mean_features(x):
    full = x.mean(1, dtype=np.float64)
    scale = np.sqrt(x.var(1, dtype=np.float64) + 1e-5)
    edge = []
    for view in (x, x[:, ::-1]):
        recent = [view[:, -n:].mean(1, dtype=np.float64) - full for n in (6, 24, 72)]
        recent += [view[:, -24:].mean(1, dtype=np.float64) - view[:, -48:-24].mean(1, dtype=np.float64)]
        edge.append(np.stack(recent, axis=-1))
    return full, scale, edge[0], edge[1]


def coefficient_family(centered, bins, mode):
    spectrum = np.fft.rfft(centered, axis=1)
    components = spectrum[:, bins, :].imag.copy()
    if mode == "remove":
        spectrum[:, bins, :] = spectrum[:, bins, :].real
        return np.fft.irfft(spectrum, n=LENGTH, axis=1)
    if mode == "reflect":
        spectrum[:, bins, :] = spectrum[:, bins, :].conj()
        return np.fft.irfft(spectrum, n=LENGTH, axis=1).astype(np.float32)
    # Periodic continuation of these sine coefficients, averaged over the requested horizon.
    omega = 2 * np.pi * bins / LENGTH
    gain = np.sin(np.arange(HORIZON)[:, None] * omega).mean(axis=0)
    return np.einsum("nbc,b->nc", components, -2 * gain / LENGTH)


def ridge_path(features, labels):
    """Fit all fixed penalties for a sample-mean square loss and standardized inputs."""
    x, y = np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    center, scale, intercept = x.mean(0), x.std(0), y.mean(0)
    scale = np.where(scale > 1e-9, scale, 1.)
    x = (x - center) / scale
    gram = x.T @ x / len(x)
    covariance = x.T @ (y - intercept) / len(x)
    eigenvalues, basis = np.linalg.eigh(gram)
    diagonal = basis.T @ covariance
    weights = np.array([basis @ (diagonal / (np.maximum(eigenvalues, 0)[:, None] + alpha))
                        for alpha in ALPHAS])
    return center, scale, intercept, weights


def choose_linear(train_x, train_y, val_x, val_y, evaluation):
    center, scale, intercept, weights = ridge_path(train_x, train_y)
    val = ((val_x - center) / scale) @ weights + intercept
    choice = ((val - val_y) ** 2).mean(axis=1).argmin(axis=0)
    columns = np.arange(len(choice))
    selected = weights[choice, :, columns].T
    estimates = {name: ((features - center) / scale) @ selected + intercept for name, features in evaluation.items()}
    return estimates, ALPHAS[choice].tolist()


def prepare_features(series, split):
    x, y = windows(series, split)
    center, scale, recent, early = mean_features(x)
    centered = x - center[:, None, :]
    future = y.mean(1, dtype=np.float64)
    target = np.stack([future - center, recent[..., 1]], axis=-1)
    return x, y, {"time": series["time"][series[split]], "origin": series[split], "center": center,
                   "scale": scale, "recent": recent, "early": early, "true_mean": future, "target": target,
                   "edge_mean": coefficient_family(centered, EDGE, "mean"),
                   "inside_mean": coefficient_family(centered, EDGE - 1, "mean")}


def raw_probe(dataset, data, records):
    matrices = {}
    for split, series in data.items():
        x, _ = windows(series, "test" if split == "external" else split)
        matrices[split] = x.transpose(0, 2, 1) - records[split]["center"][..., None]
    labels = {split: rec["target"].reshape(-1, 2) for split, rec in records.items()}
    outputs, tuning = {}, {}
    for name, bins in (("full", None), ("edge_removed", EDGE), ("lower_removed", EDGE - 1), ("upper_removed", EDGE + 1)):
        features = {}
        for split, x in matrices.items():
            if bins is not None:
                x = coefficient_family(x.transpose(0, 2, 1), bins, "remove").transpose(0, 2, 1)
            features[split] = x.reshape(-1, LENGTH)
        estimates, penalties = choose_linear(features["train"], labels["train"], features["val"], labels["val"],
                                             {k: v for k, v in features.items() if k in ("test", "external")})
        tuning[name] = penalties
        for split, estimate in estimates.items():
            outputs[f"{split}_{name}"] = estimate.reshape(records[split]["target"].shape)
        print(dataset, "raw readout", name, penalties, flush=True)
    return outputs, tuning


def load_model(checkpoints, name, variant):
    path = checkpoints / name / variant
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model = PairedPatchTST(SimpleNamespace(**config), variant).cuda().eval()
    model.load_state_dict(torch.load(path / "checkpoint.pth", weights_only=True, map_location="cpu"), strict=True)
    return model


def frozen_outputs(model, x, projection, include_forecast):
    seen = {}
    def record(name):
        def capture(_module, _inputs, result):
            seen[name] = result[0]
        return capture

    hooks = [model.patch_embedding.register_forward_hook(record("embedding")),
             model.encoder.register_forward_hook(record("encoder"))]
    results = {"embedding": [], "encoder": [], "forecast": []}
    try:
        with torch.inference_mode():
            for start in range(0, len(x), 64):
                batch = torch.as_tensor(x[start:start + 64], device="cuda")
                pred = model(batch, None, None, None)
                for name, layer in seen.items():
                    # Equal width for both tokenizers. Four pools keep coarse position; the
                    # fixed orthogonal feature projection is shared and has never seen a label.
                    vector = F.adaptive_avg_pool1d((layer @ projection).transpose(1, 2), 4)
                    results[name].append(vector.reshape(*batch.shape[::2], -1).cpu().numpy())
                if include_forecast:
                    results["forecast"].append(pred.cpu().numpy())
                else:
                    results["forecast"].append(pred.mean(1).cpu().numpy())
    finally:
        for hook in hooks:
            hook.remove()
    return {name: np.concatenate(pieces) for name, pieces in results.items()}


def neural_probe(dataset, latent, records):
    labels = {split: rec["target"].reshape(-1, 2) for split, rec in records.items()}
    outputs, tuning = {}, {}
    for name in ("embedding", "encoder", "encoder_recent", "encoder_early"):
        features = {}
        for split, state in latent.items():
            base = state["embedding" if name == "embedding" else "encoder"]
            rec = records[split]
            scale = rec["scale"][..., None].astype(np.float32)
            pieces = [base * scale, scale]
            if name in ("encoder_recent", "encoder_early"):
                pieces.append(rec[name.split("_")[1]].astype(np.float32))
            features[split] = np.concatenate(pieces, axis=-1).reshape(-1, sum(x.shape[-1] for x in pieces))
        estimates, penalties = choose_linear(features["train"], labels["train"], features["val"], labels["val"],
                                             {k: v for k, v in features.items() if k in ("test", "external")})
        tuning[name] = penalties
        for split, estimate in estimates.items():
            outputs[f"{split}_{name}"] = estimate.reshape(records[split]["target"].shape)
        print(dataset, "frozen readout", name, penalties, flush=True)
    return outputs, tuning


def calibrate(latent, records):
    val = records["val"]
    cut = int(len(val["origin"]) * .60)
    select_from = int(np.searchsorted(val["origin"], val["origin"][cut - 1] + LENGTH + HORIZON))
    if len(val["origin"]) - select_from < 100:
        raise ValueError("Insufficient validation windows after the history/target embargo")
    targets = val["true_mean"] - latent["val"]["forecast"]
    predictions, tuning = {}, {}
    for name in ("linear", "early", "recent", "history_stats", "edge_mean", "inside_mean"):
        features = {}
        for split, state in latent.items():
            if split == "train":
                continue
            rec = records[split]
            mean = state["forecast"] if split == "val" else state["forecast"].mean(1, dtype=np.float64)
            parts = [(mean - rec["center"])[..., None]]
            if name in ("early", "recent"):
                parts.append(rec[name])
            elif name == "history_stats":
                parts.extend([rec["center"][..., None], rec["scale"][..., None]])
            elif name in ("edge_mean", "inside_mean"):
                parts.append(rec[name][..., None])
            features[split] = np.concatenate(parts, axis=-1)
        val_features = features["val"]
        choices = np.zeros(len(ALPHAS))
        models = []
        for channel in range(targets.shape[1]):
            x = val_features[:, channel]
            y = targets[:, channel, None]
            stats = ridge_path(x[:cut], y[:cut])
            center, scale, intercept, weights = stats
            pred = ((x[select_from:] - center) / scale) @ weights + intercept
            choices += ((pred - y[select_from:]) ** 2).mean(axis=(1, 2))
            models.append(stats)
        chosen = int(choices.argmin())
        tuning[name] = {"alpha": float(ALPHAS[chosen]), "fit_windows": cut,
                        "selection_windows": len(val["origin"]) - select_from}
        for split in ("test", "external"):
            if split not in features:
                continue
            x = features[split]
            offset = np.column_stack([(((x[:, c] - center) / scale) @ weights[chosen] + intercept)[:, 0]
                                      for c, (center, scale, intercept, weights) in enumerate(models)])
            predictions[f"{split}_{name}"] = latent[split]["forecast"].mean(1, dtype=np.float64) + offset
    return predictions, tuning


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def intervention(model, variant, x):
    # Perturb the signs of real input sine components while keeping mean and variance fixed.
    real = {}
    with torch.inference_mode():
        base = model(torch.as_tensor(x, device="cuda"), None, None, None).cpu().numpy()
        for label, indices in (("edge", EDGE), ("lower", EDGE - 1), ("upper", EDGE + 1)):
            changed = coefficient_family(x, indices, "reflect")
            pred = model(torch.as_tensor(changed, device="cuda"), None, None, None).cpu().numpy()
            du = rms(changed - x)
            real[label] = {"input_rms": du, "forecast_rms": rms(pred - base),
                           "horizon_mean_rms": rms((pred - base).mean(1, dtype=np.float64))}

        cases = []
        sample_history = np.arange(LENGTH)
        shared = 0.8 * np.cos(2 * np.pi * 3 * sample_history / LENGTH)
        for freq in (7, 8, 9, 15, 16, 17, 23, 24, 25):
            wave = np.sin(2 * np.pi * freq * sample_history / LENGTH)
            z = np.stack([shared + wave, shared - wave]).astype(np.float32)
            signal = torch.as_tensor(z[:, :, None], device="cuda")
            norm = (signal - signal.mean(1, keepdim=True)) / (signal.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
            if variant == "timeft":
                tokens = model.patch_embedding.tokenize(norm.transpose(1, 2)).cpu().numpy()
            else:
                embed = model.patch_embedding
                tokens = embed.padding_patch_layer(norm.transpose(1, 2)).unfold(-1, embed.patch_len, embed.stride).cpu().numpy()
            prediction = model(signal, None, None, None).cpu().numpy()[:, :, 0]
            cases.append({"bin": freq, "raw_token_rms": rms(tokens[0] - tokens[1]),
                          "forecast_rms": rms(prediction[0] - prediction[1]),
                          "forecast_mean_difference": float(prediction[0].mean() - prediction[1].mean()),
                          "history_mean_difference": float(z[0].mean() - z[1].mean()),
                          "history_variance_difference": float(z[0].var() - z[1].var())})
    return {"analytic_counterfactuals": cases, "real_reflections": real, "real_windows": len(x)}


def run_dataset(args, dataset, projection, sources):
    print("START", dataset, flush=True)
    dest = args.output / dataset
    dest.mkdir(parents=True, exist_ok=False)
    series = read_npz(args.inputs / f"{dataset}.npz")
    reference = read_npz(args.inputs / f"{dataset}-reference.npz")
    splits = {split: series for split in ("train", "val", "test")}
    if dataset == "Weather":
        splits["external"] = read_npz(args.inputs / "Weather2021.npz")
    records = {}
    for split, block in splits.items():
        _, _, records[split] = prepare_features(block, "test" if split == "external" else split)
    raw, settings = raw_probe(dataset, splits, records)
    np.savez_compressed(dest / "raw_probes.npz", **raw)
    write_json(dest / "raw_settings.json", settings)
    train_shift_cut = np.quantile(np.abs(records["train"]["recent"][..., 1]), .75, axis=0)
    for split in ("test", "external"):
        if split not in splits:
            continue
        rec = records[split]
        np.savez_compressed(dest / f"{split}_truth.npz", time=rec["time"], origin=rec["origin"],
                            target=rec["target"], truth_mean=rec["true_mean"], center=rec["center"],
                            recent_shift=rec["recent"][..., 1], high_shift_cut=train_shift_cut,
                            channels=series["channels"], minutes=series["minutes"])

    for variant in VARIANTS:
        for filename, expected in sources["benchmark"][dataset]["model_artifacts"][variant].items():
            with (args.checkpoints / dataset / variant / filename).open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
        model = load_model(args.checkpoints, dataset, variant)
        latent = {}
        for split, block in splits.items():
            x, _ = windows(block, "test" if split == "external" else split)
            latent[split] = frozen_outputs(model, x, projection, split in ("test", "external"))
            print(dataset, variant, split, len(x), flush=True)
        parity = latent["test"]["forecast"][:len(reference[variant])] - reference[variant]
        if rms(parity) > 8e-4:
            raise RuntimeError(f"Checkpoint predictions disagree with saved reference: {rms(parity):.4g}")
        decoded, tuning = neural_probe(f"{dataset} {variant}", latent, records)
        calibrated, calib_tuning = calibrate(latent, records)
        result = {**decoded, **calibrated}
        for split in ("test", "external"):
            if split not in splits:
                continue
            _, y = windows(splits[split], "test")
            pred = latent[split]["forecast"]
            residual = pred.astype(np.float64) - y
            baseline_mean = pred.mean(1, dtype=np.float64)
            mean_error = residual.mean(1)
            baseline_wave_mse = ((residual - mean_error[:, None, :]) ** 2).mean(1)
            result[f"{split}_baseline"] = baseline_mean
            result[f"{split}_wave_mse"] = baseline_wave_mse
            np.save(dest / f"{variant}_{split}_pred.npy", pred, allow_pickle=False)
        x, _ = windows(series, "test")
        index = np.linspace(0, len(x)-1, min(64, len(x)), dtype=int)
        audit = intervention(model, variant, x[index])
        write_json(dest / f"{variant}_audit.json", {"reference_rms": rms(parity),
                   "reference_max_abs": float(np.abs(parity).max()), "probe_penalties": tuning,
                   "calibration": calib_tuning, **audit})
        np.savez_compressed(dest / f"{variant}_diagnostics.npz", **result)
        del model, latent
        print("FINISHED", dataset, variant, flush=True)


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen inference expects CUDA")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2021)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(1)
    basis, _ = np.linalg.qr(np.random.default_rng(2021).normal(size=(512, 48)))
    projection = torch.as_tensor(basis, device="cuda", dtype=torch.float32)
    sources = json.loads((args.inputs / "inputs.json").read_text(encoding="utf-8"))
    for dataset in args.datasets:
        run_dataset(args, dataset, projection, sources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    main(parser.parse_args())
