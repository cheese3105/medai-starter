from __future__ import annotations

import unittest

from attack.sampling import sample_pairs


class SamplingTests(unittest.TestCase):
    def test_same_seed_returns_same_pairs(self) -> None:
        first = sample_pairs(["t1", "t2"], ["e1", "e2"], 3, 42)
        second = sample_pairs(["t1", "t2"], ["e1", "e2"], 3, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_sample_is_capped_to_cartesian_product(self) -> None:
        result = sample_pairs(["t1"], ["e1", "e2"], 10, 42)
        self.assertEqual(set(result), {("t1", "e1"), ("t1", "e2")})

    def test_non_positive_sample_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sample_pairs(["t1"], ["e1"], 0, 42)


if __name__ == "__main__":
    unittest.main()

