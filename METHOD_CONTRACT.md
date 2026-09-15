# Executable method contract derived from the reviewed PDF

## Scope

The reviewed method consists of task-aware sparse communication, client/task
error feedback, energy-and-affinity scheduling, and selective freezing of the
shared extractor. TopK-SGD is a baseline, not the proposed compressor.

## Corrections required by the reviews

1. **Task-aware sparse operator.** Every task first forms a deterministic
   top-`rho` support. If task similarity exceeds `sigma`, coordinates selected
   by more than one similar task are assigned to the task with the largest
   absolute error-compensated value. This makes `rho` an upper bound in every
   regime, makes `sigma -> 1` recover ordinary per-task TopK, and matches the
   figure's claim that redundant overlaps are transmitted once.
2. **Error feedback.** A distinct residual is stored for every
   `(client, task)` pair. The identity `transmitted + next_residual =
   raw_update + previous_residual` must hold elementwise.
3. **Payload.** Sparse payload includes values, coordinate indices, task ID,
   transmitted count, and framing metadata. The experiment logger reports the
   realized representation-level record length. Network headers,
   retransmissions, acknowledgments, and downlink broadcasts require transport
   instrumentation and are not presented as measured transmitted bytes.
4. **Selective freezing.** Only the shared extractor may freeze. Task heads
   always train. To prevent cross-task cancellation, stationarity is measured
   as the weighted root-mean-square of the individual shared-gradient norms,
   not the norm of their weighted sum. Two consecutive probe checks below the
   threshold are required.
5. **Scheduling.** Communication cost is represented as the normalized benefit
   `1 - cost/Cmax`, so the equally weighted composite score remains in `[0,1]`.
   A deterministic minimum-participation
   phase is followed by feasible score-per-cost greedy selection. The reviewed
   manuscript's `(1-1/e)` claim is not retained: ordinary density greedy for
   0-1 knapsack does not have that guarantee. Approximation claims require a
   separately proved algorithm.
6. **Energy claims.** Until physical logs are supplied, energy is an analytic
   or simulator-derived estimate and must not be described as measured device
   energy. Wall-clock and process memory profiling are reported separately.

## Experimental target

- Datasets: UCI HAR, CWRU Bearing, Intel Lab Sensor Data.
- Main baselines: FedAvg, FMTL, FedProx, TopK-SGD, SCAFFOLD.
- Proposed method: the four-component method above.
- Five fixed seeds, seed-level logs, confidence intervals, convergence curves,
  task metrics, payload bytes, mask density, residual memory, freeze onset,
  selected-client frequency, and budget utilization.
