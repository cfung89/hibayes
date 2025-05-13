from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pandas as pd
import pytest
from numpyro import handlers

from hibayes.model.models import (
    BaseModel,
    BetaBinomial,
    Binomial,
    ModelConfig,
    ParameterConfig,
    PriorConfig,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    A minimal dataframe that will exercise the default BetaBinomial/Binomial
    """
    return pd.DataFrame(
        {
            "model": ["m1", "m1", "m2", "m2"],
            "task": ["t1", "t1", "t1", "t1"],
            "score": [1, 0, 1, 0],
        }
    )


@pytest.fixture
def beta_binomial_model() -> BetaBinomial:
    return BetaBinomial()


@pytest.fixture
def binomial_model() -> Binomial:
    return Binomial()


def _get_trace(model: BaseModel, df: pd.DataFrame):
    """run the model once (no inference) and return the numpyro trace."""
    features, *_ = model.prepare_data(df)
    rng_key = jax.random.PRNGKey(0)
    seeded_model = handlers.seed(model.build_model(), rng_key)
    return handlers.trace(seeded_model).get_trace(**features)


def test_prepare_data_requires_obs():
    """
    A subclass whose _prepare_data forgets to include an 'obs' feature
    must raise a ValueError from BaseModel.prepare_data.
    """

    class _BadModel(BaseModel):
        @classmethod
        def get_default_config(cls) -> ModelConfig:
            # trivial config with a single parameter so that sample_param works
            return ModelConfig(
                configurable_parameters=[
                    ParameterConfig(
                        name="alpha", prior=PriorConfig(distribution=dist.Normal)
                    )
                ]
            )

        def _prepare_data(self, data: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
            # intentionally omit 'obs' from the features
            return {}, {}, {}

        def build_model(self):
            return lambda obs=None: None

    with pytest.raises(ValueError, match="observation feature"):
        _BadModel().prepare_data(pd.DataFrame({"a": [1, 2, 3]}))


def test_betabinomial_prepare_data_shapes(
    beta_binomial_model: BetaBinomial, sample_df: pd.DataFramex
):
    """
    Ensure the automatic aggregation and factorisation logic in
    BetaBinomial._prepare_data produces the expected feature/coord/dim layout.
    """
    features, coords, dims = beta_binomial_model.prepare_data(sample_df)

    assert "obs" in features and "total_count" in features
    assert features["obs"].shape == (2,)  # one row per distinct (model, task)
    assert all(features["total_count"] == jnp.array([2, 2], dtype=jnp.int32))

    # auto-generated indexing helpers
    assert "model_effects_index" in features
    assert "num_model_effects" in features
    assert "task_effects_index" in features
    assert "num_task_effects" in features
    assert features["num_model_effects"] == 2
    assert features["num_task_effects"] == 1

    # coords / dims
    assert "model" in coords and len(coords["model"]) == 2
    assert "task" in coords and len(coords["task"]) == 1
    assert dims["model_effects"] == ["model"]
    assert dims["task_effects"] == ["task"]


# identical to above.
def test_binomial_prepare_data_shapes(
    binomial_model: Binomial, sample_df: pd.DataFramex
):
    """
    Ensure the automatic aggregation and factorisation logic in
    BetaBinomial._prepare_data produces the expected feature/coord/dim layout.
    """
    features, coords, dims = binomial_model.prepare_data(sample_df)

    assert "obs" in features and "total_count" in features
    assert features["obs"].shape == (2,)  # one row per distinct (model, task)
    assert all(features["total_count"] == jnp.array([2, 2], dtype=jnp.int32))

    # auto-generated indexing helpers
    assert "model_effects_index" in features
    assert "num_model_effects" in features
    assert "task_effects_index" in features
    assert "num_task_effects" in features
    assert features["num_model_effects"] == 2
    assert features["num_task_effects"] == 1

    # coords / dims
    assert "model" in coords and len(coords["model"]) == 2
    assert "task" in coords and len(coords["task"]) == 1
    assert dims["model_effects"] == ["model"]
    assert dims["task_effects"] == ["task"]


def test_betabinomial_model_trace_has_expected_nodes(
    beta_binomial_model: BetaBinomial, sample_df: pd.DataFrame
):
    """
    A single forward pass should create the right sample/deterministic names
    without throwing.
    """
    tr = _get_trace(beta_binomial_model, sample_df)

    # Core parameters
    assert "overall_mean" in tr
    assert "model_effects" in tr
    assert "task_effects" in tr
    assert "dispersion_phi" in tr

    # Deterministics + likelihood
    assert "p_bar" in tr
    assert "success_prob" in tr
    assert "n_correct" in tr


def test_binomial_model_trace_has_expected_nodes(
    binomial_model: Binomial,
    sample_df: pd.DataFrame,
):
    tr = _get_trace(binomial_model, sample_df)

    assert "overall_mean" in tr
    assert "model_effects" in tr
    assert "task_effects" in tr
    assert "dispersion_phi" not in tr
    assert "success_prob" in tr
    assert "n_correct" in tr
