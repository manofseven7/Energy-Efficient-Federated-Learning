"""Aggregate confirmatory, ablation, and sensitivity simulations.

Only 300-round runs with the declared client protocol are accepted.  The
script writes machine-readable tables and publication-ready PDF figures.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "analysis_extended"
SEEDS = {11, 22, 33, 44, 55}


def ci95(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))


def dataset_name(path):
    return Path(path).stem


def valid_manifest(path):
    manifest = json.loads(path.read_text())
    dataset = dataset_name(manifest["dataset"])
    expected_clients = 52 if dataset == "intel_sensor" else 80
    return (
        manifest["rounds"] == 300
        and manifest["seed"] in SEEDS
        and manifest["clients"] == expected_clients
        and manifest["clients_per_round"] == 24
    )


def load_run(manifest_path, family):
    m = json.loads(manifest_path.read_text())
    curve = pd.read_csv(manifest_path.parent / "round_metrics.csv")
    part = pd.read_csv(manifest_path.parent / "participation.csv")
    dataset = dataset_name(m["dataset"])
    counts = part["rounds"].to_numpy(float)
    jain = counts.sum() ** 2 / (len(counts) * np.square(counts).sum())
    final = curve.iloc[-1]
    def column(name, default=0.0):
        return curve[name] if name in curve else pd.Series(np.full(len(curve), default))
    return {
        "family": family,
        "run": manifest_path.parent.name,
        "dataset": dataset,
        "method": m["method"],
        "variant": m.get("variant", "full"),
        "seed": m["seed"],
        "rho": m.get("rho", 0.3),
        "sigma": m.get("sigma", 0.6),
        "eta_f": m.get("eta_f", 5e-4),
        "task1": final.task1_metric,
        "task2": final.task2_metric,
        "payload_mb": m["payload_bytes"] / 1e6,
        "residual_mb": m["residual_bytes"] / 1e6,
        "wall_seconds": m["wall_seconds"],
        "final_frozen": final.get("frozen_clients", 0.0),
        "mean_frozen": column("frozen_clients").mean(),
        "mean_similarity": column("mean_task_similarity").mean(),
        "mean_retention": column("realized_retention_fraction", 1.0).mean(),
        "mean_budget_utilization": column("round_budget_utilization").mean(),
        "mean_stationarity": column("mean_probe_stationarity").mean(),
        "min_stationarity": column("min_probe_stationarity").min(),
        "residual_l2_final": final.get("residual_l2", 0.0),
        "jain_participation": jain,
    }


def collect():
    rows = []
    sources = [
        ("main", RESULTS / "full", lambda m: json.loads(m.read_text())["method"] != "proposed"),
        ("main", RESULTS / "v2" / "main", lambda m: True),
        ("component", RESULTS / "extended" / "component", lambda m: json.loads(m.read_text()).get("variant") == "no_scheduler"),
        ("component", RESULTS / "v2" / "component", lambda m: True),
        ("rho", RESULTS / "v2" / "rho", lambda m: True),
        ("sigma", RESULTS / "v2" / "sigma", lambda m: True),
        ("eta", RESULTS / "v2" / "eta", lambda m: True),
    ]
    for family, directory, include in sources:
        for manifest in sorted(directory.glob("*/run_manifest.json")):
            if include(manifest) and valid_manifest(manifest):
                rows.append(load_run(manifest, family))
    return pd.DataFrame(rows)


def grouped_table(frame, keys):
    metrics = ["task1", "task2", "payload_mb", "final_frozen", "mean_retention",
               "mean_budget_utilization", "jain_participation", "wall_seconds"]
    rows = []
    for group, data in frame.groupby(keys, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        row = dict(zip(keys, group)); row["n"] = len(data)
        for metric in metrics:
            row[f"{metric}_mean"] = data[metric].mean()
            row[f"{metric}_sd"] = data[metric].std(ddof=1)
            row[f"{metric}_ci95"] = ci95(data[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def style():
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .2,
                         "figure.dpi": 150, "savefig.bbox": "tight"})


def error_panel(table, x, title, xlabel, path):
    style(); datasets = ["uci_har", "cwru", "intel_sensor"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    for ax, dataset in zip(axes, datasets):
        q = table[table.dataset == dataset].sort_values(x)
        ax.errorbar(q[x], q.task1_mean, yerr=q.task1_ci95, marker="o", capsize=2)
        ax.set_title(dataset.replace("_", " ").upper())
        ax.set_xlabel(xlabel); ax.set_ylabel("Task-1 MSE" if dataset == "intel_sensor" else "Task-1 accuracy")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def component_figure(table, path):
    style(); variants = ["full", "no_scheduler", "no_taskaware", "no_ef", "no_freeze"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6))
    for ax, dataset in zip(axes, ["uci_har", "cwru", "intel_sensor"]):
        q = table[table.dataset == dataset].set_index("variant").reindex(variants)
        ax.bar(np.arange(len(q)), q.task1_mean, yerr=q.task1_ci95, capsize=2)
        ax.set_xticks(np.arange(len(q)), ["Full", "No sched.", "No task", "No EF", "No freeze"], rotation=40, ha="right")
        ax.set_title(dataset.replace("_", " ").upper())
        ax.set_ylabel("MSE (lower)" if dataset == "intel_sensor" else "Accuracy")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def telemetry_figure(raw, path):
    style(); q = raw[(raw.family == "sigma")]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    for dataset, group in q.groupby("dataset"):
        s = group.groupby("sigma").mean(numeric_only=True).reset_index()
        axes[0].plot(s.sigma, s.mean_retention, marker="o", label=dataset)
        axes[1].plot(s.sigma, s.payload_mb, marker="o", label=dataset)
    axes[0].set(xlabel=r"Similarity threshold $\sigma$", ylabel="Realized TopK retention")
    axes[1].set(xlabel=r"Similarity threshold $\sigma$", ylabel="Cumulative payload (MB)")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def participation_figure(raw, path):
    style(); q = raw[raw.family == "component"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    variants = ["full", "no_scheduler"]
    labels = []
    for dataset in ["uci_har", "cwru", "intel_sensor"]:
        for variant in variants:
            values = q[(q.dataset == dataset) & (q.variant == variant)].jain_participation
            axes[0].scatter([len(labels)] * len(values), values, s=12)
            labels.append(dataset.split("_")[0] + "\n" + ("sched" if variant == "full" else "random"))
    axes[0].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axes[0].set_ylabel("Jain participation index")
    full = q[q.variant == "full"]
    for dataset, group in full.groupby("dataset"):
        axes[1].scatter(group.mean_budget_utilization, group.jain_participation, label=dataset, s=18)
    axes[1].set(xlabel="Mean round-budget utilization", ylabel="Jain participation index")
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = collect(); raw.to_csv(OUT / "extended_seed_level.csv", index=False)
    specs = {
        "main": ["dataset", "method"],
        "component": ["dataset", "variant"],
        "rho": ["dataset", "rho"],
        "sigma": ["dataset", "sigma"],
        "eta": ["dataset", "eta_f"],
    }
    tables = {}
    for family, keys in specs.items():
        tables[family] = grouped_table(raw[raw.family == family], keys)
        tables[family].to_csv(OUT / f"table_{family}.csv", index=False)
    paired = []
    main = raw[raw.family == "main"]
    for dataset in sorted(main.dataset.unique()):
        proposed = main[(main.dataset == dataset) & (main.method == "proposed")].set_index("seed")
        for method in sorted(set(main.method) - {"proposed"}):
            baseline = main[(main.dataset == dataset) & (main.method == method)].set_index("seed")
            common = proposed.index.intersection(baseline.index)
            for metric in ["task1", "task2"]:
                delta = proposed.loc[common, metric] - baseline.loc[common, metric]
                test = stats.ttest_rel(proposed.loc[common, metric], baseline.loc[common, metric])
                paired.append({"dataset": dataset, "baseline": method, "metric": metric,
                               "n": len(common), "mean_delta_proposed_minus_baseline": delta.mean(),
                               "delta_ci95": ci95(delta), "t_statistic": test.statistic,
                               "p_value_two_sided": test.pvalue})
    pd.DataFrame(paired).to_csv(OUT / "paired_tests_main.csv", index=False)
    component_figure(tables["component"], OUT / "fig_component_ablation.pdf")
    error_panel(tables["rho"], "rho", r"Retention sensitivity", r"Retention $\rho$", OUT / "fig_rho_sensitivity.pdf")
    error_panel(tables["sigma"], "sigma", r"Similarity-threshold sensitivity", r"Threshold $\sigma$", OUT / "fig_sigma_sensitivity.pdf")
    if not tables["eta"].empty:
        error_panel(tables["eta"], "eta_f", r"Freezing-threshold sensitivity", r"Threshold $\eta_f$", OUT / "fig_eta_sensitivity.pdf")
    telemetry_figure(raw, OUT / "fig_sigma_telemetry.pdf")
    participation_figure(raw, OUT / "fig_participation_budget.pdf")
    audit = {
        "accepted_runs": int(len(raw)),
        "runs_by_family": {k: int(v) for k, v in raw.family.value_counts().to_dict().items()},
        "acceptance_rule": "300 rounds; seeds 11/22/33/44/55; 80 clients (52 observed Intel sensors); 24 selected",
    }
    (OUT / "analysis_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
