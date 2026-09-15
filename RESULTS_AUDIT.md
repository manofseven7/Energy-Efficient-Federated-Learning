# Complete computational-run audit

## Execution status

- 3 datasets × 6 methods × 5 seeds = 90 runs.
- 255 additional runs cover component ablation and the declared
  \(\rho\), \(\sigma\), and \(\eta_f\) sensitivity grids.
- Every run completed 300 communication rounds.
- All final metrics are finite.
- Active clients per round: 24 for every compared method after the scheduler fix.
- These are computational/simulation results, not measurements from the
  reviewed manuscript's claimed 80-board physical testbed.

## Mean final results over five seeds

For UCI HAR and CWRU, higher is better. For Intel Sensor, values are normalized
MSE and lower is better.

| Dataset | Method | Task 1 | Task 2 | Uplink payload (MB) |
|---|---|---:|---:|---:|
| UCI HAR | FedAvg | 0.8635 | 0.9996 | 521.856 |
| UCI HAR | FMTL | 0.8621 | 0.9996 | 521.856 |
| UCI HAR | FedProx | 0.8635 | 0.9996 | 521.856 |
| UCI HAR | TopK-SGD | 0.8628 | 0.9996 | 226.901 |
| UCI HAR | SCAFFOLD | 0.8627 | 0.9994 | 521.856 |
| UCI HAR | Proposed | 0.8699 | 0.9995 | 226.901 |
| CWRU | FedAvg | 0.9360 | 0.9009 | 28.800 |
| CWRU | FMTL | 0.9345 | 0.9012 | 28.800 |
| CWRU | FedProx | 0.9360 | 0.9009 | 28.800 |
| CWRU | TopK-SGD | 0.9358 | 0.9007 | 13.032 |
| CWRU | SCAFFOLD | 0.9254 | 0.9030 | 28.800 |
| CWRU | Proposed | 0.9574 | 0.9100 | 13.032 |
| Intel Sensor | FedAvg | 0.2360 | 0.0406 | 7.430 |
| Intel Sensor | FMTL | 0.2384 | 0.0403 | 7.430 |
| Intel Sensor | FedProx | 0.2360 | 0.0406 | 7.430 |
| Intel Sensor | TopK-SGD | 0.2363 | 0.0407 | 3.269 |
| Intel Sensor | SCAFFOLD | 0.2335 | 0.0396 | 7.430 |
| Intel Sensor | Proposed | 0.2114 | 0.0369 | 3.269 |

Exact standard deviations, 95% t intervals, paired tests, seed-level values,
and round-level curves are in `results/analysis/`.

## Claims supported by these runs

- The corrected scheduler can maintain the same 24-client participation level
  as the baselines without exceeding its simulated round budget.
- Sparse communication reduces algorithmic uplink payload by approximately
  54–56% relative to dense 32-bit update transport under the implemented
  index-aware payload model.
- The proposed combination improves Task-1 mean performance on UCI HAR and
  CWRU and lowers Task-1 MSE on Intel Sensor in these runs.

## Claims not supported

- Selective freezing never activated at the default `eta_f = 5e-4`; no
  default-method computation saving may be attributed to it. Sensitivity runs
  show activation at larger thresholds together with dataset-dependent
  predictive degradation.
- At the default `sigma = 0.6`, task-aware compression did not transmit fewer
  bytes than TopK-SGD. At `sigma = 0`, overlap removal reduced payload by
  4.8--8.4% relative to independent TopK, with a small performance trade-off.
- No physical energy, Raspberry Pi, ESP32, STM32, INA219, Wi-Fi, ZigBee, or
  14-hour testbed claim is reproduced.
- The old numerical energy table and its percentage reductions cannot be
  retained as measured results.

## Remaining scientific limitations

The UCI HAR and CWRU two-task label constructions are now explicitly defined,
encoded in the preprocessing source, and described as benchmark constructions;
they are not claimed to be recovered historical assignments. Intel Sensor is
rebuilt from the official raw dataset with a chronological split. FMTL is
implemented and named as an explicit Laplacian head-coupling baseline rather
than implying an unrecovered solver from another paper. Utility scheduling
improves mean Task-1 results but reduces Jain participation equality; this is
reported as a trade-off.

## Editorial consequence

The computational study is complete and auditable, and the manuscript has been
rewritten around it. It is technically ready for an editorial pre-submission
check. Resubmission to the same review record remains high-risk because its
scope is narrower than the reviewed physical-testbed paper; editor approval of
that scope change is advisable before treating the package as final.
