"""Apply the frozen comparison rules and summarize the 2022 holdout."""

import argparse
import json
from pathlib import Path

import numpy as np

from reproduction.mechanism.summarize import VARIANTS, load_npz, paired_blocks, write_table


def evaluate(primary, followup, output):
    table, probe_rows, effects = [], [], {}
    new_truth = load_npz(followup / "2022_truth.npz")
    primary_truth = {year: load_npz(primary / "Weather" / f"{split}_truth.npz")
                     for year, split in (("2020", "test"), ("2021", "external"))}
    primary_models = {v: load_npz(primary / "Weather" / f"{v}_diagnostics.npz") for v in VARIANTS}
    new_models = {v: load_npz(followup / f"{v}.npz") for v in VARIANTS}
    for variant in VARIANTS:
        old = json.loads((primary / "Weather" / f"{variant}_audit.json").read_text(encoding="utf-8"))
        new = json.loads((followup / f"{variant}_settings.json").read_text(encoding="utf-8"))
        if new["calibration"] != old["calibration"]:
            raise ValueError(f"The 2022 study did not retain the original calibration tuning for {variant}")
    for year in ("2020", "2021", "2022"):
        old = year in primary_truth
        truth = primary_truth[year] if old else new_truth
        prefix = {"2020": "test_", "2021": "external_", "2022": "external_"}[year]
        mask = np.abs(truth["recent_shift"]) >= truth["high_shift_cut"] if old else truth["high_shift"]
        level = truth["truth_mean"] if old else truth["true_mean"]
        t = truth["time"]
        losses, curves = {}, {}
        for variant in VARIANTS:
            if old:
                state = primary_models[variant]
                wave = state[prefix + "wave_mse"]
                baseline = state[prefix + "baseline"]
            else:
                state = new_models[variant]
                wave, baseline = state["wave_mse"], state["baseline"]
            linear = state[prefix + "linear"]
            recent, early = state[prefix + "recent"], state[prefix + "early"]
            methods = {"baseline": baseline, "linear": linear, "always_recent": recent,
                       "high_shift_recent": np.where(mask, recent, linear),
                       "high_shift_early": np.where(mask, early, linear)}
            curves[variant] = wave
            for name, prediction in methods.items():
                loss = (prediction - level) ** 2
                losses[variant, name] = loss
                table.append({"year": year, "model": variant, "mean_rule": name, "windows": len(t),
                    "high_shift_fraction": float(mask.mean()), "level_mse": float(loss.mean()),
                    "total_mse": float((loss + wave).mean()), "high_shift_mean_mse": float(loss[mask].mean()),
                    "other_mean_mse": float(loss[~mask].mean())})

            error = new_models[variant][year] - truth["target"]
            probe_rows.append({"year": year, "variant": variant, "future_mean_mse": float((error[..., 0]**2).mean()),
                              "recent_mean_mse": float((error[..., 1]**2).mean())})

        gap = losses["timeft", "baseline"] - losses["patch", "baseline"]
        block = {"original_gap": paired_blocks(gap, t),
                 "original_total_gap": paired_blocks(gap + curves["timeft"] - curves["patch"], t),
                 "high_shift_gap": paired_blocks(gap, t, mask), "other_gap": paired_blocks(gap, t, ~mask)}
        for rule in ("always_recent", "high_shift_recent", "high_shift_early"):
            p_gain = losses["patch", "linear"] - losses["patch", rule]
            tf_gain = losses["timeft", "linear"] - losses["timeft", rule]
            block[rule] = {"extra_gain_timeft": paired_blocks(tf_gain - p_gain, t),
                          "patch_gain": paired_blocks(p_gain, t), "timeft_gain": paired_blocks(tf_gain, t),
                          "timeft_high_gain": paired_blocks(tf_gain, t, mask),
                          "timeft_other_gain": paired_blocks(tf_gain, t, ~mask)}
        effects[year] = block

    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "locked_gate.csv", table)
    write_table(output / "unpooled_probe.csv", probe_rows)
    raw_horizons = json.loads((followup / "natural_horizons.json").read_text(encoding="utf-8"))
    write_table(output / "natural_horizons.csv", raw_horizons)
    (output / "followup_effects.json").write_text(json.dumps(effects, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return table, probe_rows, raw_horizons


def plot(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, probes, horizons = data
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    years = ("2020", "2021", "2022")
    x = np.arange(len(years))
    w = .22
    for i, (rule, label, color) in enumerate((
            ("baseline", "Original", "#26374a"), ("linear", "Linear calibration", "#6983a0"),
            ("always_recent", "Recent at all windows", "#2b7766"), ("high_shift_recent", "Recent only at high shift", "#ba5b30"))):
        gains = [next(r["level_mse"] for r in rows if (r["year"], r["model"], r["mean_rule"]) == (y, "timeft", rule)) for y in years]
        axes[0].plot(x, gains, "o-", color=color, label=label)
    axes[0].set(title="TimeFT mean correction over years", ylabel="Future-window mean MSE", xticks=x, xticklabels=years)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")

    for i, (variant, label, color) in enumerate((("patch", "PatchTST", "#26374a"), ("timeft", "+TimeFT", "#2b7766"))):
        values = [next(r["recent_mean_mse"] for r in probes if (r["year"], r["variant"]) == (y, variant)) for y in years]
        axes[1].bar(x + (i - .5) * w, values, width=w, label=label, color=color)
    axes[1].set(title="Recent level readout, no token pooling", ylabel="Recent-level MSE", xticks=x, xticklabels=years)
    axes[1].legend(frameon=False, fontsize=8)

    closest = [r for r in horizons if r["year"] == "2022"]
    h = np.array([r["horizon"] for r in closest if r["input"] == "full"])
    full = np.array([r["mse"] for r in closest if r["input"] == "full"])
    for kind, label, color in (("edge_removed", "Band-start phase", "#ba5b30"),
                                   ("lower_removed", "Adjacent below", "#26374a"),
                                   ("upper_removed", "Adjacent above", "#2b7766")):
        values = np.array([r["mse"] for r in closest if r["input"] == kind])
        axes[2].plot(h, 100 * (values / full - 1), "o-", color=color, label=label)
    axes[2].axhline(0, linewidth=.8, color="#aaaaaa")
    axes[2].set(title="Erasing phase on natural weather, 2022", xlabel="Forecast horizon (steps)",
                ylabel="Change in mean MSE (%)", xticks=h)
    axes[2].legend(frameon=False, fontsize=8)
    fig.savefig(output / "followup.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("followup", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(evaluate(args.primary, args.followup, args.output), args.output)
