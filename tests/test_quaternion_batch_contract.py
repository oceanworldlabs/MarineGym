"""Batch-shape contract for quaternion/vector rotation helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocean_compat.math import quat_rotate, quat_rotate_inverse  # noqa: E402


class QuaternionBatchContractTests(unittest.TestCase):
    def test_matching_batch_shapes_round_trip(self):
        q = torch.tensor(
            [[2**-0.5, 0.0, 0.0, 2**-0.5], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).unsqueeze(1)
        v = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32
        ).unsqueeze(1)

        result = quat_rotate(q, v)
        self.assertEqual(result.shape, v.shape)
        torch.testing.assert_close(quat_rotate_inverse(q, result), v)

    def test_mismatched_batch_shapes_raise_value_error(self):
        cases = [
            ((1, 4), (5, 3)),
            ((2, 1, 4), (1, 2, 3)),
        ]
        for rotate in (quat_rotate, quat_rotate_inverse):
            for q_shape, v_shape in cases:
                with self.subTest(rotate=rotate.__name__, q=q_shape, v=v_shape):
                    q = torch.ones(q_shape, dtype=torch.float32)
                    v = torch.ones(v_shape, dtype=torch.float32)

                    with self.assertRaisesRegex(
                        ValueError, "batch dimensions must match"
                    ):
                        rotate(q, v)


if __name__ == "__main__":
    unittest.main()
