#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out=results/lr_schedule
mkdir -p "$out/logs"
for dataset in ETTh1 Weather; do
    data=dataset
    if [[ "$dataset" == Weather ]]; then data=results/attribution/data; fi
    for variant in patch timeft; do
        "${TIMEFT_PYTHON:-python}" -u -m reproduction.table5.train \
            --dataset "$dataset" --variant "$variant" --data-dir "$data" \
            --lradj cosine --seed 2021 --output "$out" 2>&1 | tee "$out/logs/$dataset-$variant.log"
    done
done
