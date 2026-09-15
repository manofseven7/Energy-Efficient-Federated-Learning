"""Auditable core operators for the reviewed EA-FedMTL method."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


def topk_support(vector: np.ndarray, rho: float) -> np.ndarray:
    """Return a deterministic support containing at most ceil(rho * P) entries."""
    vector = np.asarray(vector)
    if vector.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    keep = min(vector.size, max(1, int(math.ceil(rho * vector.size))))
    # lexsort makes ties reproducible by preferring the lower coordinate index.
    order = np.lexsort((np.arange(vector.size), -np.abs(vector)))
    mask = np.zeros(vector.size, dtype=bool)
    mask[order[:keep]] = True
    return mask


def cosine_similarity_matrix(vectors: Sequence[np.ndarray], delta: float = 1e-12) -> np.ndarray:
    """Compute a finite pairwise cosine-similarity matrix."""
    matrix = np.stack([np.asarray(v, dtype=np.float64) for v in vectors])
    dots = matrix @ matrix.T
    norms = np.linalg.norm(matrix, axis=1)
    denom = norms[:, None] * norms[None, :] + delta
    result = dots / denom
    np.fill_diagonal(result, 1.0)
    return np.clip(result, -1.0, 1.0)


def task_aware_masks(
    compensated: Sequence[np.ndarray], rho: float, sigma: float
) -> tuple[list[np.ndarray], np.ndarray]:
    """Construct top-rho masks and suppress redundant cross-task overlaps.

    When two tasks have similarity greater than ``sigma``, an overlapping
    coordinate is assigned to the task with the larger magnitude. Thus sigma=1
    exactly recovers independent TopK masks.
    """
    if not -1.0 <= sigma <= 1.0:
        raise ValueError("sigma must be in [-1, 1]")
    vectors = [np.asarray(v, dtype=np.float64) for v in compensated]
    if not vectors or len({v.shape for v in vectors}) != 1:
        raise ValueError("all task vectors must have one common non-empty shape")
    similarities = cosine_similarity_matrix(vectors)
    masks = [topk_support(v, rho) for v in vectors]
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            if similarities[left, right] <= sigma:
                continue
            overlap = np.flatnonzero(masks[left] & masks[right])
            for coordinate in overlap:
                lv = abs(vectors[left][coordinate])
                rv = abs(vectors[right][coordinate])
                if lv >= rv:  # deterministic tie break
                    masks[right][coordinate] = False
                else:
                    masks[left][coordinate] = False
    return masks, similarities


def compress_with_error_feedback(
    raw_updates: Sequence[np.ndarray],
    previous_residuals: Sequence[np.ndarray],
    rho: float,
    sigma: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Apply client/task-specific error feedback followed by sparse masking."""
    if len(raw_updates) != len(previous_residuals):
        raise ValueError("one residual is required for every task update")
    compensated = [
        np.asarray(update, dtype=np.float32) + np.asarray(residual, dtype=np.float32)
        for update, residual in zip(raw_updates, previous_residuals)
    ]
    masks, similarities = task_aware_masks(compensated, rho, sigma)
    transmitted = [np.where(mask, value, 0.0).astype(np.float32) for mask, value in zip(masks, compensated)]
    residuals = [(value - sent).astype(np.float32) for value, sent in zip(compensated, transmitted)]
    return transmitted, residuals, masks, similarities


def sparse_payload_bytes(mask: np.ndarray, value_bits: int = 32, index_bits: int | None = None) -> int:
    """Return the explicit algorithmic uplink payload for one sparse task vector.

    Framing is 4-byte transmitted count, 2-byte task identifier, and 1-byte
    value-width metadata. Coordinates and values are then packed at their
    declared widths.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.size == 0:
        raise ValueError("mask must be a non-empty vector")
    if value_bits <= 0:
        raise ValueError("value_bits must be positive")
    if index_bits is None:
        index_bits = max(1, int(math.ceil(math.log2(mask.size))))
    count = int(mask.sum())
    return 7 + int(math.ceil(count * (value_bits + index_bits) / 8.0))


def cancellation_safe_stationarity(shared_gradients: Sequence[np.ndarray], weights: Iterable[float]) -> float:
    """Weighted RMS gradient norm, immune to cancellation between tasks."""
    gradients = [np.asarray(g, dtype=np.float64) for g in shared_gradients]
    weights = np.asarray(list(weights), dtype=np.float64)
    if len(gradients) == 0 or len(gradients) != len(weights):
        raise ValueError("a weight is required for every task gradient")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative with positive sum")
    weights = weights / weights.sum()
    squared_norms = np.asarray([float(g @ g) for g in gradients])
    return float(np.sqrt(weights @ squared_norms))


def should_freeze_shared(current: float, previous: float | None, threshold: float) -> bool:
    """Freeze only after two consecutive valid probe checks below threshold."""
    return previous is not None and current < threshold and previous < threshold


@dataclass(frozen=True)
class Candidate:
    client_id: int
    score: float
    cost: float


def select_budgeted_clients(
    candidates: Sequence[Candidate], budget: float, minimum: int = 0,
    maximum: int | None = None,
) -> tuple[list[int], float]:
    """Deterministic feasible scheduler without an unsupported approximation claim."""
    if budget < 0 or minimum < 0 or (maximum is not None and maximum < minimum):
        raise ValueError("budget and minimum must be non-negative")
    if any(c.cost <= 0 for c in candidates):
        raise ValueError("candidate costs must be positive")
    ranked = sorted(candidates, key=lambda c: (-c.score / c.cost, -c.score, c.client_id))
    chosen: list[Candidate] = []
    spent = 0.0
    # The least-cost set is used only for the feasibility proof.  When the
    # highest-ranked floor set itself fits, seed selection with that set; this
    # avoids permanently favoring low client IDs when costs tie.  If it does
    # not fit, fall back to the least-cost feasible set.
    if minimum:
        cheapest = sorted(candidates, key=lambda c: (c.cost, c.client_id))[:minimum]
        if len(cheapest) < minimum or sum(c.cost for c in cheapest) > budget + 1e-12:
            raise ValueError("minimum participation is infeasible under the round budget")
        preferred = ranked[:minimum]
        floor_set = preferred if sum(c.cost for c in preferred) <= budget + 1e-12 else cheapest
        for candidate in floor_set:
            chosen.append(candidate)
            spent += candidate.cost
    chosen_ids = {c.client_id for c in chosen}
    for candidate in ranked:
        if maximum is not None and len(chosen) >= maximum:
            break
        if candidate.client_id in chosen_ids or candidate.score <= 0:
            continue
        if spent + candidate.cost <= budget + 1e-12:
            chosen.append(candidate)
            chosen_ids.add(candidate.client_id)
            spent += candidate.cost
    return sorted(chosen_ids), spent
