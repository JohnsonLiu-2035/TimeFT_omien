"""Attribute Table 5 paired forecast errors using the original test window ordering."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reproduction.table5.analyze import BANDS


def quartiles(values):
    groups = np.empty(len(values), dtype=int)
    for q, indices in enumerate(np.array_split(np.argsort(values, kind="stable"), 4), 1):
        groups[indices] = q
    return groups


def concentration(history):
    power = abs(np.fft.rfft(history, axis=1)[:, 1:]) ** 2
    total = power.sum(1)
    top = np.partition(power, -3, axis=1)[:, -3:].sum(1)
    return np.divide(top, total, out=np.zeros_like(total), where=total > 0).mean(1)


def intervals(values, groups, block=416):
    """Circular moving-block bootstrap; preserve channels and both models together."""
    masks = groups[:, None] == np.arange(1, 5)
    return grouped_intervals(masks, masks * values[:, None], block)


def grouped_intervals(count, total, block):
    n = len(count)

    def sums(array):
        extended = np.concatenate([array, array[:block]])
        partial = np.vstack([np.zeros((1, 4)), extended.cumsum(axis=0)])
        return partial[block : block + n] - partial[:n]

    counts, totals = sums(count), sums(total)
    starts = np.random.default_rng(2021).integers(n, size=(2000, int(np.ceil(n / block))))
    den = counts[starts].sum(1)
    scores = np.divide(totals[starts].sum(1), den, out=np.full(den.shape, np.nan), where=den > 0)
    return np.nanquantile(np.column_stack([scores, scores[:, 0] - scores[:, 3]]), [0.025, 0.975], axis=0)


def analyze(folder, out, reference):
    def load(name):
        return np.load(folder / (name + ".npy"), mmap_mode="r")

    past, target, patch, timeft = map(load, ["history", "true", "patch/pred", "timeft/pred"])
    n, h, c = target.shape
    assert patch.shape == timeft.shape == target.shape and past.shape == (n, 320, c) and h == 96
    names = json.loads((folder / "data.json").read_text())["channels"]
    weights = np.full(49, 2.0)
    weights[[0, -1]] = 1
    time = np.arange(320, dtype=float) - 159.5
    slope_weights = 319 * time / (time @ time)
    batches = {}

    def save(key, value):
        batches.setdefault(key, []).append(value)

    for i in range(0, n, 128):
        x = past[i : i + 128]
        y, p, t = (a[i : i + 128].astype(float) for a in (target, patch, timeft))
        save("conc", concentration(x))
        # Adjacent 160-step historical windows. Scores use no forecast targets.
        save("drift", abs(x[:, 160:].mean(1, dtype=float) - x[:, :160].mean(1, dtype=float)))
        save("trend", abs(np.einsum("wtc,t->wc", x, slope_weights)))
        err = (t - y) ** 2 - (p - y) ** 2
        save("step", err.mean(2))
        save("channel_delta", err.mean(1))
        dc = (t - y).mean(1) ** 2 - (p - y).mean(1) ** 2
        save("dc", dc)
        save("patch_dc", (p - y).mean(1) ** 2)
        yf, pf, tf = (np.fft.rfft(a, axis=1) / h for a in (y, p, t))
        for band, sl in BANDS.items():
            w = weights[sl, None]

            def power_band(z):
                return (w * abs(z[:, sl]) ** 2).sum(1)

            save(band, (power_band(tf - yf) - power_band(pf - yf)).mean(1))
            if band != "DC":
                save(band + "_patch_power", power_band(pf))
                save(band + "_patch_cross", (w * (pf[:, sl].conj() * yf[:, sl]).real).sum(1))
        for name, a in [("patch", p), ("timeft", t), ("true", y)]:
            save(name + "_var", a.var(1).mean(1))
            save(name + "_rough", (np.diff(a, axis=1) ** 2).mean((1, 2)))
            if name != "true":
                cov = ((a - a.mean(1, keepdims=True)) * (y - y.mean(1, keepdims=True))).mean(1)
                save(name + "_cov", cov.mean(1))
                denom = np.sqrt(a.var(1) * y.var(1))
                save(
                    name + "_corr",
                    np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 1e-12),
                )
                save(name + "_mse", ((a - y) ** 2).mean((1, 2)))
    a = {k: np.concatenate(v) for k, v in batches.items()}
    group = quartiles(a["conc"])
    # A stronger smoothing-only comparator for Q1: permit a different fixed
    # attenuation for every channel and every non-DC Fourier coefficient.
    indices = np.flatnonzero(group == 1)

    def moments(offsets):
        energy = np.zeros((48, c))
        cross = np.zeros_like(energy)
        for i in range(0, len(offsets), 128):
            idx = offsets[i : i + 128]
            pf, yf = (np.fft.rfft(arr[idx].astype(float), axis=1)[:, 1:] / h for arr in (patch, target))
            energy += (abs(pf) ** 2).sum(0)
            cross += (pf.conj() * yf).real.sum(0)
        return energy, cross

    def shrink_change(fit, score):
        energy, cross = fit
        gains = np.clip(np.divide(cross, energy, out=np.ones_like(energy), where=energy > 0), 0, 1)
        e, s = score
        return weights[1:, None] * ((gains**2 - 1) * e - 2 * (gains - 1) * s)

    full = moments(indices)
    bin_delta = shrink_change(full, full) / len(indices)
    heldout = np.zeros_like(bin_delta)
    for fold in np.array_split(np.arange(n), 5):
        test = indices[(indices >= fold[0]) & (indices <= fold[-1])]
        train = indices[(indices <= fold[0] - 416) | (indices >= fold[-1] + 416)]
        assert len(train)
        heldout += shrink_change(moments(train), moments(test)) / len(indices)
    step = a["step"]
    delta = step.mean(1)
    level = a["dc"].mean(1)
    shape = delta - level
    variance = a["timeft_var"] - a["patch_var"]
    alignment = -2 * (a["timeft_cov"] - a["patch_cov"])
    np.testing.assert_allclose(shape, variance + alignment, atol=1e-12)
    np.testing.assert_allclose(delta, sum(a[b] for b in BANDS), atol=1e-12)
    for metric in ("patch", "timeft"):
        np.testing.assert_allclose(a[metric + "_mse"].mean(), reference[metric]["mse"], atol=1e-8)
    records, horizons, uncertainty = [], [], []
    for label, values in [("total", delta), ("dc", level), ("shape", shape)]:
        ci = intervals(values, group)
        for j, key in enumerate(["Q1", "Q2", "Q3", "Q4", "Q1-Q4"]):
            uncertainty.append(dict(metric=label, group=key, low=ci[0, j], high=ci[1, j]))
    for q in range(1, 5):
        mask = group == q
        row = dict(
            q=q,
            n=mask.sum(),
            concentration=a["conc"][mask].mean(),
            delta=delta[mask].mean(),
            shape=shape[mask].mean(),
            variance=variance[mask].mean(),
            alignment=alignment[mask].mean(),
        )
        for k in [
            *BANDS,
            "patch_var",
            "timeft_var",
            "true_var",
            "patch_rough",
            "timeft_rough",
            "true_rough",
            "patch_mse",
            "timeft_mse",
        ]:
            row[k] = a[k][mask].mean()
        for m in ("patch", "timeft"):
            np.testing.assert_allclose(
                row[m + "_mse"], reference["periodicity_quartiles"][q - 1][m + "_mse"], atol=1e-6
            )
        valid = np.isfinite(a["patch_corr"][mask]) & np.isfinite(a["timeft_corr"][mask])
        row["corr_coverage"] = valid.mean()
        for m in ("patch", "timeft"):
            row[m + "_corr"] = a[m + "_corr"][mask][valid].mean()
        # Oracle: one real gain in [0,1] per band and channel, fitted and scored
        # on the same quartile. An optimistic ceiling for this shrinkage family.
        shrink_delta = 0.0
        for b in ("low", "mid", "high"):
            energy = a[b + "_patch_power"][mask].mean(0)
            cross = a[b + "_patch_cross"][mask].mean(0)
            g = np.clip(np.divide(cross, energy, out=np.ones(c), where=energy > 0), 0, 1)
            shrink_delta += ((g * g - 1) * energy - 2 * (g - 1) * cross).mean()
        row["oracle_shrink_delta"] = shrink_delta
        records.append(row)
        horizons.extend(dict(q=q, lead=j + 1, delta=v) for j, v in enumerate(step[mask].mean(0)))
    drift = a["drift"]
    rows = []
    channel_score = drift.mean(0)
    count = np.zeros((n, 4))
    dc_sum = np.zeros((n, 4))
    for j, name in enumerate(names):
        groups = quartiles(drift[:, j])
        for q in range(1, 5):
            mask = groups == q
            count[:, q - 1] += mask
            dc_sum[:, q - 1] += mask * a["dc"][:, j]
            rows.append(
                dict(
                    channel=name,
                    channel_drift=channel_score[j],
                    channel_q=quartiles(channel_score)[j],
                    window_q=q,
                    n=mask.sum(),
                    drift=drift[mask, j].mean(),
                    patch_dc=a["patch_dc"][mask, j].mean(),
                    dc=a["dc"][mask, j].mean(),
                    timeft_wins=(a["dc"][mask, j] < 0).mean(),
                    cut_min=drift[mask, j].min(),
                    cut_max=drift[mask, j].max(),
                )
            )
    ci = grouped_intervals(count, dc_sum, 416)
    for j, key in enumerate(["Q1", "Q2", "Q3", "Q4", "Q1-Q4"]):
        uncertainty.append(
            dict(metric="within_channel_drift_dc", group=key, low=ci[0, j], high=ci[1, j])
        )
    out.mkdir(exist_ok=True, parents=True)
    trend_groups = np.column_stack([quartiles(series) for series in a["trend"].T])
    sensitivity = []
    for q in range(1, 5):
        mask = trend_groups == q
        sensitivity.append(dict(q=q, trend=a["trend"][mask].mean(), dc=a["dc"][mask].mean()))
    pd.DataFrame(sensitivity).to_csv(out / "trend.csv", index=False)
    block_sensitivity = []
    for block in (96, 416, 832):
        for metric, values in (("total", delta), ("shape", shape)):
            bounds = intervals(values, group, block)[:, -1]
            block_sensitivity.append(dict(block=block, metric=metric, low=bounds[0], high=bounds[1]))
    pd.DataFrame(block_sensitivity).to_csv(out / "q1_q4_sensitivity.csv", index=False)
    pd.DataFrame(
        dict(
            channel=names,
            delta=a["channel_delta"][indices].mean(0),
            dc=a["dc"][indices].mean(0),
            oracle_bin_shrink_delta=bin_delta.sum(0),
            heldout_bin_shrink_delta=heldout.sum(0),
        )
    ).to_csv(out / "q1_channels.csv", index=False)
    for name, table in [
        ("concentration", records),
        ("horizon", horizons),
        ("drift", rows),
        ("intervals", uncertainty),
    ]:
        pd.DataFrame(table).to_csv(out / (name + ".csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/table5/formal"))
    parser.add_argument("--output", type=Path, default=Path("reproduction/attribution/report"))
    args = parser.parse_args()
    ref = json.loads(Path("reproduction/table5/report/analysis/metrics.json").read_text())
    for data in ref:
        analyze(args.root / data, args.output / data, ref[data])
