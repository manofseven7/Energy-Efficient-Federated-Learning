#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

run_one() {
  local data="$1" output="$2" variant="$3" rho="$4" sigma="$5" eta="$6" seed="$7"
  python -m src.experiment --data "$data" --method proposed \
    --variant "$variant" --rho "$rho" --sigma "$sigma" --eta-f "$eta" \
    --seed "$seed" --rounds 300 --clients 80 --clients-per-round 24 \
    --output "$output"
}

for spec in \
  "uci_har:data/processed/uci_har.npz" \
  "cwru:data/processed/cwru.npz" \
  "intel_sensor:data/processed/intel_sensor.npz"
do
  name="${spec%%:*}"; data="${spec#*:}"
  for variant in full no_scheduler no_taskaware no_ef no_freeze
  do
    for seed in 11 22 33 44 55
    do
      if [ "$variant" = no_scheduler ]; then base="results/extended/component"; else base="results/v2/component"; fi
      run_one "$data" "$base/${name}_${variant}_${seed}" "$variant" 0.3 0.6 0.0005 "$seed"
    done
  done
  for rho in 0.1 0.3 0.5
  do
    for seed in 11 22 33 44 55
    do run_one "$data" "results/v2/rho/${name}_rho${rho}_${seed}" full "$rho" 0.6 0.0005 "$seed"
    done
  done
  for sigma in 0.0 0.3 0.6 0.9 1.0
  do
    for seed in 11 22 33 44 55
    do run_one "$data" "results/v2/sigma/${name}_sigma${sigma}_${seed}" full 0.3 "$sigma" 0.0005 "$seed"
    done
  done
  for eta in 0.0005 0.1 0.5 1.0
  do
    for seed in 11 22 33 44 55
    do run_one "$data" "results/v2/eta/${name}_eta${eta}_${seed}" full 0.3 0.6 "$eta" "$seed"
    done
  done
done
