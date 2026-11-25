"""Analyze paired predictions without retraining or treating windows as independent."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PAPER = {
    "ETTh1": ((0.403, 0.423), (0.376, 0.392)),
    "ETTm1": ((0.321, 0.361), (0.290, 0.339)),
    "Weather": ((0.155, 0.204), (0.147, 0.194)),
}
BANDS = {"DC": slice(0, 1), "low": slice(1, 5), "mid": slice(5, 13), "high": slice(13, 49)}


def timeft_roundtrip(x):
    n, stride, width = 320, 20, 8
    spectrum = np.fft.rfft(x, axis=-1)
    band_count = (spectrum.shape[-1] + width - 1) // width
    padded = np.pad(spectrum, ((0, 0), (0, 0), (0, band_count * width - spectrum.shape[-1])))
    subbands = np.fft.irfft(padded.reshape(*x.shape[:-1], band_count, width), n=n, axis=-1)
    phases = subbands.reshape(*x.shape[:-1], band_count, n // stride, stride).swapaxes(-1, -2)
    reassembled = phases.swapaxes(-1, -2).reshape(subbands.shape)
    restored_spectrum = np.fft.rfft(reassembled, axis=-1)[..., :width].reshape(padded.shape)
    restored_spectrum = restored_spectrum[..., :spectrum.shape[-1]]
    return phases, np.fft.irfft(restored_spectrum, n=n, axis=-1), spectrum, restored_spectrum


def token_retention(history, output):
    # Use the same uniformly spaced real windows for both transformations.
    index = np.unique(np.linspace(0, len(history) - 1, min(256, len(history)), dtype=int))
    original = np.asarray(history[index], dtype=np.float64).transpose(0, 2, 1)
    centered = original - original.mean(axis=-1, keepdims=True)
    x = centered / np.sqrt(centered.var(axis=-1, keepdims=True) + 1e-5)
    n, stride = 320, 20
    patches = x.reshape(*x.shape[:-1], n // stride, stride)
    patch_reconstruction = patches.reshape(x.shape)

    phases, timeft_reconstruction, spectrum, restored_spectrum = timeft_roundtrip(x)

    # Extract representative intermediates without storing the same history for each model.
    take = min(12, len(index))
    np.savez_compressed(output, window_offset=index[:take], normalized_input=x[:take],
                        patches=patches[:take], timeft_phases=phases[:take],
                        timeft_reconstruction=timeft_reconstruction[:take])
    # One grid-aligned sinusoid, deliberately not grid-aligned within a 20-sample patch.
    frequency_bin = 13
    sine = np.sin(2 * np.pi * frequency_bin * np.arange(n) / n)
    masked = np.zeros(n)
    masked[:stride] = sine[:stride]
    full_energy = np.abs(np.fft.rfft(sine)) ** 2
    patch_energy = np.abs(np.fft.rfft(masked)) ** 2
    boundary_sine = np.sin(2 * np.pi * 8 * np.arange(n) / n)
    boundary_cosine = np.cos(2 * np.pi * 8 * np.arange(n) / n)
    cases = np.stack([sine, boundary_sine, boundary_cosine])[:, None, :]
    _, recovered_cases, _, _ = timeft_roundtrip(cases)
    np.savez_compressed(output.with_name("sine_spectra.npz"), full_power=full_energy,
                        patch_power=patch_energy, reference_bin=frequency_bin)
    return {"windows": len(index), "patch_all_tokens_mae": float(np.abs(x - patch_reconstruction).mean()),
            "patch_single_token_mean_mae": float(np.abs(x).mean() * (1 - stride / n)),
            "timeft_all_tokens_mae": float(np.abs(x - timeft_reconstruction).mean()),
            "timeft_relative_spectral_error": float(np.linalg.norm(spectrum - restored_spectrum) / np.linalg.norm(spectrum)),
            "sine_full_off_bin_fraction": float(1 - full_energy[frequency_bin] / full_energy.sum()),
            "sine_patch_off_bin_fraction": float(1 - patch_energy[frequency_bin] / patch_energy.sum()),
            "sine_timeft_shifted_bin": frequency_bin % 8,
            "timeft_test_signal_mae": dict(zip(("sine_in_band", "sine_at_band_start", "cosine_at_band_start"),
                                                np.abs(cases - recovered_cases).mean(axis=(1, 2)).tolist()))}


def forecast_spectrum(pred, truth):
    horizon = truth.shape[1]
    weights = np.full(horizon // 2 + 1, 2.0)
    weights[[0, -1]] = 1.0
    pooled = np.zeros_like(weights)
    amplitude_error = dominant_weight = phase_error = phase_weight = 0.0
    horizon_sq = np.zeros(horizon)
    channel_sq = np.zeros(truth.shape[2])
    window_mse = []
    mae = 0.0
    for start in range(0, len(truth), 256):
        y = np.asarray(truth[start:start + 256], dtype=np.float64)
        p = np.asarray(pred[start:start + 256], dtype=np.float64)
        residual = p - y
        sq = residual ** 2
        horizon_sq += sq.sum(axis=(0, 2))
        channel_sq += sq.sum(axis=(0, 1))
        window_mse.append(sq.mean(axis=(1, 2)))
        mae += np.abs(residual).sum()
        yf = np.fft.rfft(y, axis=1, norm="ortho")
        pf = np.fft.rfft(p, axis=1, norm="ortho")
        pooled += (np.abs(pf - yf) ** 2).sum(axis=(0, 2)) * weights
        index = np.abs(yf[:, 1:-1]).argmax(axis=1, keepdims=True) + 1
        dominant_y = np.take_along_axis(yf, index, axis=1)
        dominant_p = np.take_along_axis(pf, index, axis=1)
        amp_y, amp_p = np.abs(dominant_y), np.abs(dominant_p)
        amplitude_error += np.abs(amp_y - amp_p).sum()
        dominant_weight += amp_y.sum()
        defined = (amp_y > 1e-10) & (amp_p > 1e-10)
        phase_error += (amp_y[defined] * np.abs(np.angle(dominant_p[defined] * dominant_y[defined].conj()))).sum()
        phase_weight += amp_y[defined].sum()
    pooled /= truth.size
    mse = horizon_sq.sum() / truth.size
    np.testing.assert_allclose(mse, pooled.sum(), rtol=1e-10, atol=1e-12)
    return {"mse": mse, "mae": mae / truth.size, "rmse": np.sqrt(mse),
            "band_mse": {key: float(pooled[sl].sum()) for key, sl in BANDS.items()},
            "dominant_relative_amplitude_error": amplitude_error / dominant_weight,
            "dominant_weighted_phase_error_degrees": float(np.rad2deg(phase_error / phase_weight)),
            "dominant_phase_weight_coverage": phase_weight / dominant_weight,
            "channel_mse": (channel_sq / (len(truth) * horizon)).tolist(),
            "horizon_mse": (horizon_sq / (len(truth) * truth.shape[2])).tolist()}, np.concatenate(window_mse)


def periodicity_comparison(history, patch_loss, timeft_loss):
    values = []
    for start in range(0, len(history), 256):
        power = np.abs(np.fft.rfft(history[start:start + 256], axis=1)[:, 1:]) ** 2
        total = power.sum(axis=1)
        top = np.partition(power, -3, axis=1)[:, -3:].sum(axis=1)
        share = np.divide(top, total, out=np.zeros_like(top), where=total > 0)
        values.append(share.mean(axis=1))
    scores = np.concatenate(values)
    bins = np.array_split(np.argsort(scores, kind="stable"), 4)
    return [{"quartile": i, "windows": len(idx), "mean_spectral_concentration": float(scores[idx].mean()),
             "patch_mse": float(patch_loss[idx].mean()), "timeft_mse": float(timeft_loss[idx].mean())}
            for i, idx in enumerate(bins, 1)]


def analyze(root):
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    rows, by_dataset = [], {}
    for dataset, quoted in PAPER.items():
        folder = root / "formal" / dataset
        if not all((folder / v / "summary.json").is_file() for v in ("patch", "timeft")):
            continue
        history = np.load(folder / "history.npy", mmap_mode="r")
        truth = np.load(folder / "true.npy", mmap_mode="r")
        result = {"channel_names": json.loads((folder / "data.json").read_text())["channels"]}
        losses = {}
        for variant, paper in zip(("patch", "timeft"), quoted):
            trial = folder / variant
            pred = np.load(trial / "pred.npy", mmap_mode="r")
            assert pred.shape == truth.shape and len(history) == len(truth)
            record = json.loads((trial / "summary.json").read_text())
            config = json.loads((trial / "config.json").read_text())
            result[variant], losses[variant] = forecast_spectrum(pred, truth)
            computed = result[variant]
            np.testing.assert_allclose([record["mse"], record["mae"]], [computed["mse"], computed["mae"]], rtol=1e-10)
            rows.append({"dataset": dataset, "variant": variant, "paper_mse": paper[0], "paper_mae": paper[1],
                         "mse": record["mse"], "mae": record["mae"], "mse_minus_paper": record["mse"] - paper[0],
                         "mae_minus_paper": record["mae"] - paper[1], "best_epoch": record["best_epoch"],
                         "epochs": record["epochs"], "tokens": config["tokens"], "parameters": config["parameters"],
                         "train_seconds": record["train_wall_seconds"], "test_seconds": record["test_wall_seconds"],
                         "peak_mib": record["peak_training_allocated_mib"]})
        dataset_analysis = analysis / dataset
        dataset_analysis.mkdir(exist_ok=True)
        result["retention"] = token_retention(history, dataset_analysis / "token_samples.npz")
        result["periodicity_quartiles"] = periodicity_comparison(history, losses["patch"], losses["timeft"])
        by_dataset[dataset] = result
    if rows:
        with (analysis / "table5.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        (analysis / "metrics.json").write_text(json.dumps(by_dataset, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return rows, by_dataset


def plot(rows, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    colors = ("#1a574b", "#96b9ad", "#96380d", "#e2ab8d")
    x = np.arange(len(names))
    for index, metric in enumerate(("mse", "mae")):
        actual = [100 * (1 - results[d]["timeft"][metric] / results[d]["patch"][metric]) for d in names]
        paper = [100 * (1 - PAPER[d][1][index] / PAPER[d][0][index]) for d in names]
        for order, (values, label) in enumerate(((actual, "observed"), (paper, "paper"))):
            offset = 2 * index + order
            axs[0].bar(x + (offset - 1.5) * 0.18, values, width=0.17, color=colors[offset], label=f"{metric.upper()} {label}")
    axs[0].set(title="Prediction error reduction", ylabel="Relative change (%)", xticks=x, xticklabels=names)
    axs[0].axhline(0, color="#888888", linewidth=0.7)
    axs[0].legend(frameon=False, fontsize=8)

    for index, name in enumerate(names):
        patch = results[name]["patch"]["band_mse"]
        timeft = results[name]["timeft"]["band_mse"]
        axs[1].bar(np.arange(len(BANDS)) + 0.2 * (index - (len(names) - 1) / 2),
                   [100 * (patch[key] - timeft[key]) / results[name]["patch"]["mse"] for key in BANDS],
                   width=0.18, label=name)
    axs[1].set(title="Where MSE changes", ylabel="% of baseline total MSE", xticks=np.arange(len(BANDS)), xticklabels=BANDS)
    axs[1].axhline(0, color="#888888", linewidth=0.7)
    axs[1].legend(frameon=False, fontsize=8)

    for index, (variant, label) in enumerate((("patch", "PatchTST"), ("timeft", "+TimeFT"))):
        seconds = [next(row["train_seconds"] for row in rows if row["dataset"] == name and row["variant"] == variant) / 60 for name in names]
        axs[2].bar(x + (index - 0.5) * 0.28, seconds, width=0.25, color=colors[index * 2], label=label)
    axs[2].set(title="Training and validation cost", ylabel="Wall-clock minutes", xticks=x, xticklabels=names)
    axs[2].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_examples(root, results):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, len(results), figsize=(4.8 * len(results), 6), squeeze=False,
                            layout="constrained")
    for index, (name, metrics) in enumerate(results.items()):
        folder = root / "formal" / name
        truth = np.load(folder / "true.npy", mmap_mode="r")
        past = np.load(folder / "history.npy", mmap_mode="r")
        midpoint = len(truth) // 2
        row = int(np.load(folder / "target_row.npy")[midpoint])
        t = np.arange(1, 97)
        axis = axs[0, index]
        axis.plot(t - 96, past[midpoint, -96:, -1], color="#777777", label="History")
        axis.plot(t, truth[midpoint, :, -1], color="#101010", label="Truth")
        costs = axs[1, index]
        for variant, label, color in (("patch", "PatchTST", "#96380d"), ("timeft", "+TimeFT", "#1a574b")):
            values = np.load(folder / variant / "pred.npy", mmap_mode="r")
            axis.plot(t, values[midpoint, :, -1], color=color, alpha=0.9, label=label)
            costs.plot(t, metrics[variant]["horizon_mse"], color=color, label=label)
        axis.axvline(0, color="#aaaaaa", linewidth=0.8)
        axis.set(title=f"{name} · middle window at row {row}", ylabel="Standardized OT")
        costs.set(title="MSE by lead time (full test set)", xlabel="Lead (steps)", ylabel="MSE")
        if index == 0:
            axis.legend(frameon=False, fontsize=8)
            costs.legend(frameon=False, fontsize=8)
    fig.savefig(root / "analysis" / "forecasts.png", dpi=180)
    plt.close(fig)


def plot_retention(root, results):
    import matplotlib.pyplot as plt

    names = list(results)
    spectra = np.load(root / "analysis" / names[0] / "sine_spectra.npz")
    fig, (freq, recovery) = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")
    bins = np.arange(len(spectra["full_power"]))
    patch = spectra["patch_power"]
    freq.plot(bins, patch / patch.sum(), color="#96380d", label="One 20-sample patch")
    freq.vlines(int(spectra["reference_bin"]), 0, 1, color="#1a574b", label="Original 320 samples")
    freq.set(title="Rectangular-window leakage", xlabel="320-point frequency bin", ylabel="Fraction of power", xlim=(0, 65))
    freq.legend(frameon=False, fontsize=8)
    x = np.arange(len(names))
    for i, (key, label, color) in enumerate((
        ("patch_single_token_mean_mae", "One patch (mean over positions)", "#96380d"),
        ("timeft_all_tokens_mae", "All TimeFT tokens", "#1a574b"),
        ("patch_all_tokens_mae", "All patches", "#777777"),
    )):
        recovery.bar(x + (i - 1) * 0.24, [results[n]["retention"][key] for n in names],
                     width=0.23, color=color, label=label)
    recovery.set(title="Reconstruction before projection", ylabel="MAE (normalized input)",
                 xticks=x, xticklabels=names, ylim=(0, 0.95))
    recovery.legend(frameon=False, fontsize=8)
    fig.savefig(root / "analysis" / "retention.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows, results = analyze(args.results)
    if not rows:
        raise SystemExit("No complete formal pairs")
    plot(rows, results, args.results / "analysis" / "overview.png")
    plot_examples(args.results, results)
    plot_retention(args.results, results)
