#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

stage=${1:?Provide smoke or formal}
dataset=${2:?Provide ETTh1, ETTm1 or Weather}
python=${TIMEFT_PYTHON:-python}
output=${TIMEFT_OUTPUT:-results/table5}
case "$stage" in
    smoke) smoke=(--smoke) ;;
    formal) smoke=() ;;
    *) exit 2 ;;
esac
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$output/logs"

for variant in patch timeft; do
    if [[ -f "$output/$stage/$dataset/$variant/summary.json" ]]; then
        printf 'Verified output already present for %s %s %s\n' "$stage" "$dataset" "$variant"
        continue
    fi
    "$python" -u -m reproduction.table5.train --dataset "$dataset" --variant "$variant" \
        --output "$output" "${smoke[@]}" 2>&1 | tee "$output/logs/$stage-$dataset-$variant.log"
done
touch "$output/$stage-$dataset.ok"
