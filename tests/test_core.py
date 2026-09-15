import unittest

import numpy as np

from src.core import (
    Candidate,
    cancellation_safe_stationarity,
    compress_with_error_feedback,
    select_budgeted_clients,
    should_freeze_shared,
    sparse_payload_bytes,
    task_aware_masks,
    topk_support,
)


class CoreContractTests(unittest.TestCase):
    def test_topk_respects_rho_and_ties_are_deterministic(self):
        mask = topk_support(np.array([2.0, -2.0, 1.0, 0.0]), 0.5)
        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_sigma_one_recovers_independent_topk(self):
        vectors = [np.array([4.0, 3.0, 1.0, 0.0]), np.array([4.0, 2.0, 0.0, 1.0])]
        masks, _ = task_aware_masks(vectors, rho=0.5, sigma=1.0)
        np.testing.assert_array_equal(masks[0], topk_support(vectors[0], 0.5))
        np.testing.assert_array_equal(masks[1], topk_support(vectors[1], 0.5))

    def test_similar_tasks_do_not_duplicate_overlapping_coordinates(self):
        vectors = [np.array([4.0, 3.0, 0.0]), np.array([4.0, 2.0, 0.0])]
        masks, similarities = task_aware_masks(vectors, rho=2 / 3, sigma=0.5)
        self.assertGreater(similarities[0, 1], 0.5)
        self.assertFalse(bool((masks[0] & masks[1]).any()))

    def test_error_feedback_conservation_for_every_task(self):
        raw = [np.array([3.0, 2.0, 1.0]), np.array([2.5, 2.0, -1.0])]
        old = [np.array([0.1, 0.2, 0.3]), np.array([-0.1, 0.0, 0.1])]
        sent, new, _, _ = compress_with_error_feedback(raw, old, 2 / 3, 0.5)
        for update, residual, transmitted, next_residual in zip(raw, old, sent, new):
            np.testing.assert_allclose(transmitted + next_residual, update + residual)

    def test_payload_includes_indices_and_metadata(self):
        mask = np.zeros(100, dtype=bool); mask[:30] = True
        self.assertEqual(sparse_payload_bytes(mask), 7 + int(np.ceil(30 * (32 + 7) / 8)))

    def test_stationarity_does_not_hide_opposing_task_gradients(self):
        value = cancellation_safe_stationarity([np.array([1.0, 0.0]), np.array([-1.0, 0.0])], [0.5, 0.5])
        self.assertAlmostEqual(value, 1.0)

    def test_freeze_requires_two_consecutive_checks(self):
        self.assertFalse(should_freeze_shared(0.01, None, 0.1))
        self.assertTrue(should_freeze_shared(0.01, 0.02, 0.1))

    def test_scheduler_never_exceeds_budget(self):
        candidates = [Candidate(0, 10, 6), Candidate(1, 8, 4), Candidate(2, -2, 1)]
        selected, spent = select_budgeted_clients(candidates, budget=10, minimum=0)
        self.assertLessEqual(spent, 10)
        self.assertNotIn(2, selected)

    def test_scheduler_rejects_infeasible_floor(self):
        candidates = [Candidate(0, 1, 6), Candidate(1, 1, 7)]
        with self.assertRaises(ValueError):
            select_budgeted_clients(candidates, budget=10, minimum=2)

    def test_scheduler_respects_cardinality_cap(self):
        candidates = [Candidate(i, 10 - i, 1) for i in range(6)]
        selected, _ = select_budgeted_clients(candidates, budget=10, maximum=3)
        self.assertEqual(len(selected), 3)

    def test_scheduler_floor_prefers_utility_when_costs_tie(self):
        candidates = [Candidate(i, float(i), 1.0) for i in range(6)]
        selected, _ = select_budgeted_clients(candidates, budget=3, minimum=2, maximum=3)
        self.assertEqual(selected, [3, 4, 5])


if __name__ == "__main__":
    unittest.main()
