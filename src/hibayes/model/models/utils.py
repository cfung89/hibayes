import copy
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Type, TypeVar, get_args, get_origin, get_type_hints

import jax.numpy as jnp

T = TypeVar("T")


def merge_sequence(
    base_seq: Sequence, update_seq: Sequence, field_type: Type
) -> Sequence:
    """Merge two sequences, with update_seq taking precedence"""
    # For lists, we'll use the update sequence by default
    # This behavior could be customized based on your specific needs
    return copy.deepcopy(update_seq)


def logit_to_prob(x):
    """Convert logit values to probabilities."""
    return 1.0 / (1.0 + jnp.exp(-x))
