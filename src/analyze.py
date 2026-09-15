"""Aggregate seed-level runs and generate manuscript-safe tables and figures."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def read_rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def dataset_from_name(name: str) -> str:
    if name.startswith("intel_sensor"):
        return "intel_sensor"
    if name.startswith("uci_har"):
        return "uci_har"
    return "cwru"


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze(results: Path, output: Path, proposed_results: Path | None = None):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    curves = []
    manifests = sorted(results.glob("*/run_manifest.json"))
    if proposed_results is not None:
        manifests = [p for p in manifests if json.loads(p.read_text())["method"] != "proposed"]
        manifests += sorted(proposed_results.glob("*/run_manifest.json"))
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        run_rows = read_rows(manifest_path.parent / "round_metrics.csv")
        final = run_rows[-1]
        dataset = dataset_from_name(manifest_path.parent.name)
        seed = int(manifest["seed"])
        records.append({
            "dataset": dataset, "method": manifest["method"], "seed": seed,
            "task1_metric": float(final["task1_metric"]),
            "task2_metric": float(final["task2_metric"]),
            "payload_bytes": int(manifest["payload_bytes"]),
            "active_clients": int(float(final["active_clients"])),
            "frozen_clients": int(float(final["frozen_clients"])),
            "wall_seconds": float(manifest["wall_seconds"]),
        })
        for row in run_rows:
            curves.append({
                "dataset": dataset, "method": manifest["method"], "seed": seed,
                "round": int(row["round"]), "task1_metric": float(row["task1_metric"]),
                "task2_metric": float(row["task2_metric"]),
            })
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["dataset"], record["method"])].append(record)
    aggregate = []
    for (dataset, method), group in sorted(grouped.items()):
        row = {"dataset": dataset, "method": method, "n_seeds": len(group)}
        for key in ("task1_metric", "task2_metric", "payload_bytes", "active_clients", "frozen_clients", "wall_seconds"):
            values = np.asarray([item[key] for item in group], dtype=float)
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_sd"] = float(values.std(ddof=1))
            row[f"{key}_ci95"] = float(stats.t.ppf(0.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
        aggregate.append(row)
    paired = []
    for dataset in sorted({record["dataset"] for record in records}):
        proposed = {r["seed"]: r for r in records if r["dataset"] == dataset and r["method"] == "proposed"}
        for method in sorted({r["method"] for r in records if r["dataset"] == dataset and r["method"] != "proposed"}):
            comparator = {r["seed"]: r for r in records if r["dataset"] == dataset and r["method"] == method}
            seeds = sorted(set(proposed) & set(comparator))
            for metric in ("task1_metric", "task2_metric"):
                left = np.asarray([proposed[s][metric] for s in seeds])
                right = np.asarray([comparator[s][metric] for s in seeds])
                test = stats.ttest_rel(left, right)
                paired.append({
                    "dataset": dataset, "comparator": method, "metric": metric,
                    "mean_difference_proposed_minus_comparator": float(np.mean(left-right)),
                    "t_statistic": float(test.statistic), "p_value_two_sided": float(test.pvalue),
                    "n_pairs": len(seeds),
                })
    write_csv(output / "seed_level_results.csv", records)
    write_csv(output / "aggregate_results.csv", aggregate)
    write_csv(output / "paired_tests.csv", paired)
    write_csv(output / "all_curves.csv", curves)

    methods = ["fedavg", "fmtl", "fedprox", "topksgd", "scaffold", "proposed"]
    datasets = ["uci_har", "cwru", "intel_sensor"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, dataset in zip(axes, datasets):
        rows = {(r["method"]): r for r in aggregate if r["dataset"] == dataset}
        values = [rows[m]["task1_metric_mean"] for m in methods]
        errors = [rows[m]["task1_metric_ci95"] for m in methods]
        axis.bar(range(len(methods)), values, yerr=errors, color=["#9aa0a6"]*5+["#c62828"], capsize=2)
        axis.set_xticks(range(len(methods)), ["FedAvg", "FMTL", "FedProx", "TopK", "SCAFFOLD", "Proposed"], rotation=45, ha="right")
        axis.set_title(dataset.replace("_", " ").upper())
        axis.set_ylabel("MSE (lower is better)" if dataset == "intel_sensor" else "Accuracy")
    fig.tight_layout(); fig.savefig(output / "fig_task1_performance.pdf", bbox_inches="tight"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    width = 0.13; x = np.arange(len(datasets))
    for index, method in enumerate(methods):
        rows = {(r["dataset"]): r for r in aggregate if r["method"] == method}
        axis.bar(x + (index-2.5)*width, [rows[d]["payload_bytes_mean"]/1e6 for d in datasets], width, label=method)
    axis.set_yscale("log"); axis.set_ylabel("Cumulative uplink payload (MB, log scale)")
    axis.set_xticks(x, ["UCI HAR", "CWRU", "Intel Sensor"]); axis.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(output / "fig_payload.pdf", bbox_inches="tight"); plt.close(fig)

    audit = {
        "run_count": len(records), "expected_run_count": 90,
        "all_finite": bool(all(np.isfinite([r["task1_metric"], r["task2_metric"]]).all() for r in records)),
        "all_300_rounds": bool(all(json.loads(p.read_text())["rounds"] == 300 for p in manifests)),
        "freezing_activated_any_run": bool(any(r["frozen_clients"] > 0 for r in records if r["method"] == "proposed")),
        "proposed_payload_below_topk": {
            d: next(r for r in aggregate if r["dataset"] == d and r["method"] == "proposed")["payload_bytes_mean"]
               < next(r for r in aggregate if r["dataset"] == d and r["method"] == "topksgd")["payload_bytes_mean"]
            for d in datasets
        },
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--results", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposed-results", type=Path)
    args = parser.parse_args(); print(json.dumps(analyze(args.results, args.output, args.proposed_results), indent=2))
