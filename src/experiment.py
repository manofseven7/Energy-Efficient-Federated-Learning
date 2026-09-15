"""Deterministic federated simulation runner with auditable outputs.

Runs of three rounds or fewer are integration smoke tests. Confirmatory and
sensitivity experiments use the declared client population, 300 rounds, five
prespecified seeds, and processed public datasets; the manifest records the
run kind so smoke-test outputs cannot be mixed into manuscript analyses.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np

from .core import (
    Candidate,
    cancellation_safe_stationarity,
    compress_with_error_feedback,
    select_budgeted_clients,
    sparse_payload_bytes,
    topk_support,
)


@dataclass
class Layout:
    features: int
    hidden: int
    classes: tuple[int, int]
    task_types: tuple[str, str] = ("classification", "classification")

    def __post_init__(self):
        sizes = [self.features * self.hidden, self.hidden]
        for count in self.classes:
            sizes.extend([self.hidden * count, count])
        self.slices = []
        start = 0
        for size in sizes:
            self.slices.append(slice(start, start + size))
            start += size
        self.parameters = start
        self.shared = np.arange(self.slices[0].start, self.slices[1].stop)
        self.task_indices = []
        for task in range(2):
            head_weight, head_bias = self.slices[2 + 2 * task:4 + 2 * task]
            self.task_indices.append(np.r_[self.shared, np.arange(head_weight.start, head_bias.stop)])

    def unpack(self, vector):
        shared_w = vector[self.slices[0]].reshape(self.features, self.hidden)
        shared_b = vector[self.slices[1]]
        heads = []
        for task, count in enumerate(self.classes):
            weight, bias = self.slices[2 + 2 * task:4 + 2 * task]
            heads.append((vector[weight].reshape(self.hidden, count), vector[bias]))
        return shared_w, shared_b, heads


def initialise(layout: Layout, rng: np.random.Generator) -> np.ndarray:
    vector = np.zeros(layout.parameters, dtype=np.float32)
    vector[layout.slices[0]] = rng.normal(0, math.sqrt(2 / (layout.features + layout.hidden)), layout.slices[0].stop)
    for task, count in enumerate(layout.classes):
        weight = layout.slices[2 + 2 * task]
        vector[weight] = rng.normal(0, math.sqrt(2 / (layout.hidden + count)), weight.stop - weight.start)
    return vector


def loss_gradient(vector, layout: Layout, x, y, task: int):
    shared_w, shared_b, heads = layout.unpack(vector)
    hidden = np.tanh(x @ shared_w + shared_b)
    head_w, head_b = heads[task]
    logits = hidden @ head_w + head_b
    if layout.task_types[task] == "regression":
        target = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        error = logits - target
        loss = float(np.mean(error * error))
        delta = (2.0 / len(x)) * error
        grad_head_w = hidden.T @ delta
        grad_head_b = delta.sum(axis=0)
        grad_hidden = (delta @ head_w.T) * (1 - hidden * hidden)
        gradient = np.zeros_like(vector)
        gradient[layout.slices[0]] = (x.T @ grad_hidden).ravel()
        gradient[layout.slices[1]] = grad_hidden.sum(axis=0)
        gradient[layout.slices[2 + 2 * task]] = grad_head_w.ravel()
        gradient[layout.slices[3 + 2 * task]] = grad_head_b
        return loss, gradient
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits); probabilities /= probabilities.sum(axis=1, keepdims=True)
    count = len(x)
    loss = float(-np.log(probabilities[np.arange(count), y] + 1e-12).mean())
    delta = probabilities.copy(); delta[np.arange(count), y] -= 1; delta /= count
    grad_head_w = hidden.T @ delta
    grad_head_b = delta.sum(axis=0)
    grad_hidden = (delta @ head_w.T) * (1 - hidden * hidden)
    gradient = np.zeros_like(vector)
    gradient[layout.slices[0]] = (x.T @ grad_hidden).ravel()
    gradient[layout.slices[1]] = grad_hidden.sum(axis=0)
    gradient[layout.slices[2 + 2 * task]] = grad_head_w.ravel()
    gradient[layout.slices[3 + 2 * task]] = grad_head_b
    return loss, gradient


def partition(labels, clients, alpha, rng):
    buckets = [[] for _ in range(clients)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label); rng.shuffle(indices)
        portions = rng.dirichlet(np.full(clients, alpha))
        for bucket, values in zip(buckets, np.split(indices, (np.cumsum(portions)[:-1] * len(indices)).astype(int))):
            bucket.extend(values.tolist())
    for index, bucket in enumerate(buckets):
        if not bucket:
            donor = max(range(clients), key=lambda other: len(buckets[other]))
            bucket.append(buckets[donor].pop())
    return [np.asarray(bucket, dtype=np.int64) for bucket in buckets]


def local_update(global_model, layout, x, labels, indices, task, rng, steps, batch_size, lr, proximal=0.0, freeze_shared=False, control=None, fmtl_strength=0.0):
    local = global_model.copy()
    active = layout.task_indices[task]
    if freeze_shared:
        active = np.setdiff1d(active, layout.shared, assume_unique=True)
    last_gradient = np.zeros_like(global_model)
    for _ in range(steps):
        chosen = rng.choice(indices, size=min(batch_size, len(indices)), replace=len(indices) < batch_size)
        _, gradient = loss_gradient(local, layout, x[chosen], labels[chosen], task)
        if proximal:
            gradient[active] += proximal * (local[active] - global_model[active])
        if control is not None:
            gradient[active] += control[active]
        # Auditable multi-task Laplacian surrogate: pull each task head toward
        # the mean of the overlapping head coordinates. Unequal-size output
        # layers are coupled only on their common prefix.
        if fmtl_strength:
            this_w = layout.slices[2 + 2 * task]
            other_w = layout.slices[2 + 2 * (1 - task)]
            overlap = min(this_w.stop - this_w.start, other_w.stop - other_w.start)
            gradient[this_w.start:this_w.start + overlap] += fmtl_strength * (
                local[this_w.start:this_w.start + overlap] - local[other_w.start:other_w.start + overlap]
            )
        local[active] -= lr * gradient[active]
        last_gradient = gradient
    return global_model - local, last_gradient


def metric(model, layout, x, labels, task):
    shared_w, shared_b, heads = layout.unpack(model)
    hidden = np.tanh(x @ shared_w + shared_b)
    head_w, head_b = heads[task]
    prediction = hidden @ head_w + head_b
    if layout.task_types[task] == "regression":
        return float(np.mean((prediction.ravel() - labels) ** 2))
    return float(np.mean(np.argmax(prediction, axis=1) == labels))


def _write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(dataset_path: Path, method: str, seed: int, rounds: int, clients: int, clients_per_round: int, output: Path,
        variant: str = "full", rho: float = 0.3, sigma: float = 0.6, eta_f: float = 5e-4):
    data = np.load(dataset_path)
    x_train = data["x_train"].astype(np.float32)
    x_test = data["x_test"].astype(np.float32)
    task_types = tuple(str(value) for value in data["task_type"].tolist()) if "task_type" in data else ("classification", "classification")
    train_labels = [data["y1_train"], data["y2_train"]]
    test_labels = [data["y1_test"], data["y2_test"]]
    train_labels = [labels.astype(np.float32) if task_types[i] == "regression" else labels.astype(int) for i, labels in enumerate(train_labels)]
    test_labels = [labels.astype(np.float32) if task_types[i] == "regression" else labels.astype(int) for i, labels in enumerate(test_labels)]
    rng = np.random.default_rng(seed)
    outputs = tuple(1 if task_types[i] == "regression" else int(labels.max() + 1) for i, labels in enumerate(train_labels))
    layout = Layout(x_train.shape[1], 16, outputs, task_types)
    model = initialise(layout, rng)
    if "client_train" in data:
        identifiers = data["client_train"]
        unique = np.unique(identifiers)
        chosen_ids = unique[:clients]
        client_parts = [np.flatnonzero(identifiers == identifier) for identifier in chosen_ids]
        clients = len(client_parts)
        clients_per_round = min(clients_per_round, clients)
    else:
        client_parts = partition(train_labels[0], clients, 0.3, rng)
    residuals = np.zeros((clients, 2, layout.parameters), dtype=np.float32)
    previous_stationarity = np.full(clients, np.nan)
    frozen = np.zeros(clients, dtype=bool)
    remaining_energy = rng.uniform(0.8, 1.2, clients)
    last_norms = np.ones((clients, 2), dtype=float)
    participation = np.zeros(clients, dtype=int)
    client_controls = np.zeros((clients, 2, layout.parameters), dtype=np.float32)
    global_controls = np.zeros((2, layout.parameters), dtype=np.float32)
    rows = []
    total_payload = 0
    start = time.perf_counter()
    for round_index in range(1, rounds + 1):
        if method == "proposed" and variant != "no_scheduler":
            costs = 0.02 + 0.005 * (1.0 - frozen.astype(float))
            affinities = last_norms.mean(axis=1) / (last_norms.mean(axis=1).max() + 1e-12)
            # All three components are normalized benefits in [0, 1]. This
            # removes the reviewed equation's negative-score ambiguity while
            # preserving equal one-third weighting: reserve, affinity, and
            # communication frugality.
            scores = (
                np.clip(remaining_energy / 1.2, 0.0, 1.0)
                + np.clip(affinities, 0.0, 1.0)
                + (1.0 - costs / costs.max())
            ) / 3.0
            floor = min(8, clients_per_round)
            selected, _ = select_budgeted_clients(
                [Candidate(i, float(scores[i]), float(costs[i])) for i in range(clients)],
                budget=float(np.sort(costs)[:clients_per_round].sum() * 1.05), minimum=floor,
                maximum=clients_per_round,
            )
        else:
            selected = rng.choice(clients, clients_per_round, replace=False).tolist()
        weighted_updates = [[], []]
        weights = []
        round_payload = 0
        round_similarities = []
        round_nnz = 0
        round_sparse_capacity = 0
        round_stationarity = []
        scheduler_budget = float(np.sort(0.02 + 0.005 * (1.0 - frozen.astype(float)))[:clients_per_round].sum() * 1.05) if method == "proposed" else 0.0
        scheduler_spent = 0.0
        for client in selected:
            participation[client] += 1
            probe_gradients = []
            probe_count = min(64, len(client_parts[client]))
            probe_indices = client_parts[client][:probe_count]
            for task in range(2):
                _, probe_gradient = loss_gradient(model, layout, x_train[probe_indices], train_labels[task][probe_indices], task)
                probe_gradients.append(probe_gradient[layout.shared])
            stationarity = cancellation_safe_stationarity(probe_gradients, [0.5, 0.5])
            round_stationarity.append(stationarity)
            if method == "proposed" and variant != "no_freeze":
                previous = None if np.isnan(previous_stationarity[client]) else previous_stationarity[client]
                frozen[client] = frozen[client] or (stationarity < eta_f and previous is not None and previous < eta_f)
            previous_stationarity[client] = stationarity
            raw_updates, gradients, control_deltas = [], [], []
            for task in range(2):
                correction = global_controls[task] - client_controls[client, task] if method == "scaffold" else None
                update, gradient = local_update(
                    model, layout, x_train, train_labels[task], client_parts[client], task, rng,
                    steps=2, batch_size=32, lr=0.01,
                    proximal=0.01 if method == "fedprox" else 0.0,
                    freeze_shared=bool(frozen[client]) if method == "proposed" else False,
                    control=correction,
                    fmtl_strength=0.01 if method == "fmtl" else 0.0,
                )
                raw_updates.append(update); gradients.append(gradient)
                last_norms[client, task] = np.linalg.norm(gradient)
                if method == "scaffold":
                    active = layout.task_indices[task]
                    updated_control = client_controls[client, task].copy()
                    updated_control[active] = (
                        client_controls[client, task, active] - global_controls[task, active]
                        + update[active] / (2 * 0.01)
                    )
                    control_deltas.append(updated_control - client_controls[client, task])
                    client_controls[client, task] = updated_control
            if method == "proposed":
                previous_residuals = residuals[client] if variant != "no_ef" else np.zeros_like(residuals[client])
                effective_sigma = 1.0 if variant == "no_taskaware" else sigma
                sent, next_residuals, masks, similarities = compress_with_error_feedback(raw_updates, previous_residuals, rho, effective_sigma)
                residuals[client] = next_residuals if variant != "no_ef" else 0.0
                task_updates = sent
                round_payload += sum(sparse_payload_bytes(mask) for mask in masks)
                round_nnz += sum(int(mask.sum()) for mask in masks)
                round_sparse_capacity += sum(int(math.ceil(rho * mask.size)) for mask in masks)
                if similarities.shape[0] > 1:
                    round_similarities.append(float(similarities[0, 1]))
            elif method == "topksgd":
                task_updates = []
                for task, update in enumerate(raw_updates):
                    compensated = update + residuals[client, task]
                    mask = topk_support(compensated, 0.3)
                    sent = np.where(mask, compensated, 0).astype(np.float32)
                    residuals[client, task] = compensated - sent
                    task_updates.append(sent); round_payload += sparse_payload_bytes(mask)
            else:
                task_updates = raw_updates
                round_payload += sum(len(layout.task_indices[task]) * 4 for task in range(2))
            for task in range(2):
                weighted_updates[task].append(task_updates[task])
                if method == "scaffold":
                    global_controls[task] += control_deltas[task] / clients
            weights.append(len(client_parts[client]))
            remaining_energy[client] = max(0.0, remaining_energy[client] - 0.02 - 0.005 * (not frozen[client]))
            if method == "proposed":
                scheduler_spent += 0.02 + 0.005 * (not frozen[client])
        normalised = np.asarray(weights, dtype=float); normalised /= normalised.sum()
        aggregates = [sum(weight * update for weight, update in zip(normalised, weighted_updates[task])) for task in range(2)]
        model -= 0.5 * (aggregates[0] + aggregates[1])
        total_payload += round_payload
        rows.append({
            "round": round_index,
            "task1_metric": metric(model, layout, x_test, test_labels[0], 0),
            "task2_metric": metric(model, layout, x_test, test_labels[1], 1),
            "round_payload_bytes": round_payload,
            "cumulative_payload_bytes": total_payload,
            "active_clients": len(selected),
            "frozen_clients": int(frozen.sum()),
            "budget_utilization": float(np.mean(1.0 - remaining_energy / 1.2)),
            "round_budget_utilization": scheduler_spent / scheduler_budget if scheduler_budget else 0.0,
            "mean_task_similarity": float(np.mean(round_similarities)) if round_similarities else 0.0,
            "realized_retention_fraction": round_nnz / round_sparse_capacity if round_sparse_capacity else 1.0,
            "mean_probe_stationarity": float(np.mean(round_stationarity)),
            "min_probe_stationarity": float(np.min(round_stationarity)),
            "residual_l2": float(np.linalg.norm(residuals)),
        })
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "round_metrics.csv", rows)
    _write_csv(output / "participation.csv", [{"client": i, "rounds": int(value)} for i, value in enumerate(participation)])
    manifest = {
        "dataset": str(dataset_path), "method": method, "variant": variant, "seed": seed, "rounds": rounds,
        "clients": clients, "clients_per_round": clients_per_round, "task_types": task_types,
        "parameters": layout.parameters, "payload_bytes": total_payload,
        "residual_bytes": int(residuals.nbytes), "wall_seconds": time.perf_counter() - start,
        "rho": rho, "sigma": sigma, "eta_f": eta_f,
        "run_kind": "smoke_test" if rounds <= 3 else "full_simulation",
        "smoke_only": bool(rounds <= 3),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--method", choices=["fedavg", "fedprox", "fmtl", "topksgd", "scaffold", "proposed"], required=True)
    parser.add_argument("--variant", choices=["full", "no_scheduler", "no_taskaware", "no_ef", "no_freeze"], default="full")
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--eta-f", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--clients-per-round", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.method, args.seed, args.rounds, args.clients, args.clients_per_round, args.output,
                         args.variant, args.rho, args.sigma, args.eta_f), indent=2))


if __name__ == "__main__":
    main()
