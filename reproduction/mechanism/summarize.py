"""Summarize the frozen experiments and compute the task-specific phase-loss bound."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DATASETS = ("Weather", "ETTh1", "ETTm1")
VARIANTS = ("patch", "timeft")
FORECASTS = ("baseline", "linear", "early", "recent", "history_stats", "edge_mean", "inside_mean")
STAGES = ("embedding", "encoder", "encoder_recent", "encoder_early")
RAW = ("full", "edge_removed", "lower_removed", "upper_removed")
WEEK_MINUTES = 7 * 1440


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def paired_blocks(values, time, mask=None):
    if mask is None:
        mask = np.ones_like(values, dtype=bool)
    keys = (time - time[0]) // WEEK_MINUTES
    _, group = np.unique(keys, return_inverse=True)
    count = np.bincount(group, mask.sum(1))
    total = np.bincount(group, np.where(mask, values, 0).sum(1))
    present = count > 0
    count, total = count[present], total[present]
    generator = np.random.default_rng(2021)
    selection = generator.integers(0, len(count), size=(4000, len(count)))
    replicates = total[selection].sum(1) / count[selection].sum(1)
    low, high = np.quantile(replicates, [.025, .975])
    return {"estimate": float(total.sum() / count.sum()), "low": float(low), "high": float(high),
            "blocks": len(count), "window_channels": int(count.sum())}


def geometry(output):
    edges = np.arange(8, 160, 8)
    h = np.arange(1, 721)
    carrier = np.sin(2 * np.pi * np.arange(h[-1])[:, None] * edges / 320)
    gains = carrier.cumsum(0) / h[:, None]
    max_bound = 2 * (gains ** 2).sum(1)
    remainder = h % 40
    mismatched_pairs = np.minimum(np.maximum(remainder - 1, 0), 40 - remainder)
    closed_form = 40 * mismatched_pairs / (2 * h.astype(float) ** 2)
    np.testing.assert_allclose(max_bound, closed_form, atol=1e-12, rtol=1e-11)
    if not np.all(max_bound[39::40] < 1e-25):
        raise AssertionError("Blind modes must integrate to zero over complete 40-step periods")
    rows = [{"horizon": int(k), "unit_rms_worst_case_mse": float(v), "unit_amplitude_bin8_mse": float(g ** 2)}
            for k, v, g in zip(h, max_bound, gains[:, 0])]
    write_table(output / "phase_bound.csv", rows)
    selected = {str(k): rows[k - 1] for k in (24, 40, 48, 80, 96, 120, 192, 336, 720)}
    # Two erased directions of the same unit RMS. Only their projection onto the
    # future-mean functional distinguishes their forecasting consequences.
    strongest = gains[95] / np.linalg.norm(gains[95])
    first = np.eye(len(edges))[0]
    orthogonal = first - strongest * (first @ strongest)
    orthogonal /= np.linalg.norm(orthogonal)
    directions = np.stack([strongest, orthogonal])
    modes = np.sqrt(2) * np.sin(2 * np.pi * np.arange(416)[:, None] * edges / 320)
    wave = directions @ modes.T
    np.testing.assert_allclose(np.sqrt((wave[:, :320] ** 2).mean(1)), 1., atol=1e-12)
    np.testing.assert_allclose(wave[:, 320:].mean(1) ** 2, [max_bound[95], 0.], atol=1e-12)
    np.savez_compressed(output / "blind_witnesses.npz", waves=wave, bin_indices=edges,
                        coefficients=directions, max_bound=max_bound[95])
    return (h, max_bound), selected


def summarize(remote, output):
    output.mkdir(parents=True, exist_ok=True)
    tables = {name: [] for name in ("baseline", "calibration", "frozen_probe", "raw_probe", "monthly", "channels", "interventions")}
    evidence = {}
    for dataset in DATASETS:
        folder = remote / dataset
        raw = load_npz(folder / "raw_probes.npz")
        models = {v: load_npz(folder / f"{v}_diagnostics.npz") for v in VARIANTS}
        audits = {v: json.loads((folder / f"{v}_audit.json").read_text(encoding="utf-8")) for v in VARIANTS}
        evidence[dataset] = {"audit": audits, "evaluation": {}}
        for variant, audit in audits.items():
            for item in audit["analytic_counterfactuals"]:
                tables["interventions"].append({"dataset": dataset, "variant": variant, **item})
        tasks = [("test", dataset, False)]
        if dataset == "Weather":
            tasks += [("external", "Weather2021", False), ("external", "Weather2021 Nov-Dec", True)]
        for split, period, matched_season in tasks:
            truth = load_npz(folder / f"{split}_truth.npz")
            keep = np.ones(len(truth["time"]), dtype=bool)
            if matched_season:
                keep = truth["time"].astype("datetime64[m]") >= np.datetime64("2021-11-01")
            t = truth["time"][keep]
            y = truth["truth_mean"][keep]
            target = truth["target"][keep]
            recency = np.abs(truth["recent_shift"][keep]) >= truth["high_shift_cut"]
            level, wave = {}, {}
            for variant, arrays in models.items():
                wave[variant] = arrays[f"{split}_wave_mse"][keep]
                for name in FORECASTS:
                    values = arrays[f"{split}_{name}"][keep]
                    loss = (values - y) ** 2
                    level[variant, name] = loss
                    tables["calibration"].append({"period": period, "variant": variant, "mean_head": name,
                                                  "level_mse": float(loss.mean()), "total_mse": float((loss + wave[variant]).mean()),
                                                  "high_shift_level_mse": float(loss[recency].mean()),
                                                  "other_level_mse": float(loss[~recency].mean())})
                for name in STAGES:
                    diff = arrays[f"{split}_{name}"][keep] - target
                    tables["frozen_probe"].append({"period": period, "variant": variant, "probe": name,
                                                   "future_mean_mse": float((diff[..., 0] ** 2).mean()),
                                                   "recent_mean_mse": float((diff[..., 1] ** 2).mean())})
            for control in RAW:
                diff = raw[f"{split}_{control}"][keep] - target
                tables["raw_probe"].append({"period": period, "input": control,
                                            "future_mean_mse": float((diff[..., 0] ** 2).mean()),
                                            "recent_mean_mse": float((diff[..., 1] ** 2).mean())})

            base = level["patch", "baseline"]
            comparison = level["timeft", "baseline"]
            baseline_gap = comparison - base
            wave_gap = wave["timeft"] - wave["patch"]
            interaction = ((level["timeft", "linear"] - level["timeft", "recent"])
                           - (level["patch", "linear"] - level["patch", "recent"]))
            late_vs_early = ((level["timeft", "early"] - level["timeft", "recent"])
                             - (level["patch", "early"] - level["patch", "recent"]))
            evaluation = {
                "windows": len(t), "timeft_minus_patch_level": paired_blocks(baseline_gap, t),
                "timeft_minus_patch_total": paired_blocks(baseline_gap + wave_gap, t),
                "high_shift": paired_blocks(baseline_gap, t, recency),
                "other_shift": paired_blocks(baseline_gap, t, ~recency),
                "additional_recent_gain_timeft_minus_patch": paired_blocks(interaction, t),
                "late_vs_early_gain_timeft_minus_patch": paired_blocks(late_vs_early, t),
            }
            evidence[dataset]["evaluation"][period] = evaluation
            tables["baseline"].append({"period": period, "windows": len(t), "patch_level": float(base.mean()),
                "timeft_level": float(comparison.mean()), "patch_wave": float(wave["patch"].mean()),
                "timeft_wave": float(wave["timeft"].mean()), "patch_total": float((base + wave["patch"]).mean()),
                "timeft_total": float((comparison + wave["timeft"]).mean()),
                "level_delta_low": evaluation["timeft_minus_patch_level"]["low"],
                "level_delta_high": evaluation["timeft_minus_patch_level"]["high"]})
            for channel, name in enumerate(truth["channels"].tolist()):
                tables["channels"].append({"period": period, "channel": name, "patch_level_mse": float(base[:, channel].mean()),
                                           "timeft_level_mse": float(comparison[:, channel].mean()),
                                           "difference": float(baseline_gap[:, channel].mean())})
            months = t.astype("datetime64[m]").astype("datetime64[M]")
            for month in np.unique(months):
                idx = months == month
                tables["monthly"].append({"period": period, "month": str(month), "windows": int(idx.sum()),
                    "level_delta": float(baseline_gap[idx].mean()), "centered_wave_delta": float(wave_gap[idx].mean())})

    for name, table in tables.items():
        write_table(output / f"{name}.csv", table)
    curve, selected = geometry(output)
    (output / "evidence.json").write_text(json.dumps({"dataset": evidence, "periodic_bound": selected},
                                                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return tables, evidence, curve


def plot(output, tables, evidence, curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), layout="constrained")
    h, bound = curve
    axes[0].plot(h[31:193], bound[31:193], color="#16554a", linewidth=1.4)
    for k in (40, 80, 120, 160):
        axes[0].axvline(k, linewidth=0.8, color="#cbd0d5")
    axes[0].scatter([96], [bound[95]], color="#b34b24", marker="o", s=25, zorder=4)
    axes[0].annotate(f"H=96: {bound[95]:.3g}", (96, bound[95]), xytext=(111, 0.058),
                     arrowprops={"arrowstyle": "-", "color": "#b34b24"}, fontsize=9)
    axes[0].set(title="Error forced by indistinguishable inputs", xlabel="Prediction horizon (steps)",
                ylabel="Worst-case mean MSE bound", ylim=(-.002, 1.10 * bound[31:193].max()), xlim=(32, 193))

    baseline = tables["baseline"]
    names = [row["period"].replace("Weather2021 Nov-Dec", "Weather 2021\nNov–Dec").replace("Weather2021", "Weather 2021")
             for row in baseline]
    indices = np.arange(len(names))
    levels = np.array([row["timeft_level"] - row["patch_level"] for row in baseline])
    waves = [row["timeft_wave"] - row["patch_wave"] for row in baseline]
    limits = np.array([[row["level_delta_low"], row["level_delta_high"]] for row in baseline]).T
    axes[1].bar(indices - .15, levels, width=.28, color="#b34b24", label="Window mean")
    axes[1].errorbar(indices - .15, levels, yerr=np.abs(limits - levels), fmt="none", color="#56220d", capsize=3)
    axes[1].bar(indices + .15, waves, width=.28, color="#16554a", label="Centered shape")
    axes[1].axhline(0, color="#777777", lw=.65)
    axes[1].set(title="Where the forecast gap appears", ylabel="MSE difference (TimeFT − PatchTST)",
                xticks=indices, xticklabels=names)
    axes[1].tick_params(axis="x", labelrotation=18, labelsize=8)
    axes[1].legend(frameon=False, fontsize=8)

    patches = []
    for name in ("Weather", "Weather2021"):
        for variant in VARIANTS:
            subset = {row["mean_head"]: row["level_mse"] for row in tables["calibration"]
                      if row["period"] == name and row["variant"] == variant}
            patches.append([subset[n] for n in ("baseline", "linear", "early", "recent")])
    controls = np.arange(4)
    for series, marker, color, style, label in zip(patches, ("o", "o", "^", "^"), ("#16554a", "#b34b24") * 2,
             ("-", "-", "--", "--"), ("Patch 2020", "TimeFT 2020", "Patch 2021", "TimeFT 2021")):
        axes[2].plot(controls, series, marker=marker, color=color, linestyle=style, label=label)
    axes[2].set(title="Adding information to the frozen mean", ylabel="Window-mean MSE", xticks=controls,
                xticklabels=["Original", "Scale + bias", "Early stats", "Recent stats"])
    axes[2].tick_params(axis="x", labelrotation=15, labelsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    fig.savefig(output / "overview.png", dpi=170)
    plt.close(fig)

    witnesses = load_npz(output / "blind_witnesses.npz")["waves"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True, layout="constrained")
    steps = np.arange(-64, 96)
    for axis, wave, title in zip(axes, witnesses, ("Changes the 96-step mean", "Leaves the 96-step mean unchanged")):
        y = wave[256:]
        axis.plot(steps, y, color="#16554a", lw=1, label="Positive phase combination")
        axis.plot(steps, -y, color="#b34b24", lw=1, alpha=.75, label="Flipped phase combination")
        axis.hlines([y[64:].mean(), -y[64:].mean()], 0, 95, color=["#16554a", "#b34b24"], linestyles="dashed")
        axis.axvline(0, color="#555555", lw=.8)
        axis.set(title=title, ylabel="Signal value")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].set_xlabel("Steps relative to the prediction origin")
    fig.savefig(output / "blind_witnesses.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("remote", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.output, *summarize(args.remote, args.output))
