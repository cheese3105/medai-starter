"""Deterministic sampling helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")


def sample_pairs(
    targets: Sequence[T],
    injected_examples: Sequence[U],
    sample_size: int,
    seed: int,
) -> list[tuple[T, U]]:
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    pairs = [(target, injected) for target in targets for injected in injected_examples]
    random.Random(seed).shuffle(pairs)
    return pairs[: min(sample_size, len(pairs))]

