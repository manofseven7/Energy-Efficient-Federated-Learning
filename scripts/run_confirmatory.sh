#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

for spec in \
  "uci_har:data/processed/uci_har.npz" \
  "cwru:data/processed/cwru.npz" \
  "intel_sensor:data/processed/intel_sensor.npz"
do
  name="${spec%%:*}"
  data="${spec#*:}"
  for method in fedavg fedprox fmtl topksgd scaffold proposed
  do
    for seed in 11 22 33 44 55
    do
      if [ "$method" = proposed ]; then
        out="results/v2/main/${name}_${method}_${seed}"
      else
        out="results/full/${name}_${method}_${seed}"
      fi
      python -m src.experiment --data "$data" --method "$method" \
        --variant full --rho 0.3 --sigma 0.6 --eta-f 0.0005 \
        --seed "$seed" --rounds 300 --clients 80 \
        --clients-per-round 24 --output "$out"
    done
  done
done
