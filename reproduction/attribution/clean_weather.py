"""Prepare the Weather cleaning-only control without changing the split or timestamps."""

import argparse
from pathlib import Path

import pandas as pd


def prepare(source, output):
    frame = pd.read_csv(source, float_precision="round_trip")
    values = frame.drop(columns="date")
    missing = values <= -9990
    train_end = int(0.7 * len(frame))
    assert missing.iloc[:train_end].any(axis=1).sum() == 80
    assert not missing.iloc[train_end:].any().any()
    for name in values.columns[missing.any()]:
        bad = missing[name]
        filled = (
            values[name].iloc[:train_end].mask(bad.iloc[:train_end]).interpolate(limit_area="inside")
        )
        frame.loc[bad, name] = filled.loc[bad.iloc[:train_end]]
    assert not frame.isna().any().any()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
