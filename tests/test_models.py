from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp
import numpyro.distributions as dist
import pandas as pd
import pytest

from hibayes.analysis_state import ModelAnalysisState
from hibayes.model import fit
from hibayes.model.models import (
    BaseModel,
    BetaBinomial,
    Binomial,
    ModelConfig,
    ParameterConfig,
    PriorConfig,
)


def _fit_model(model: BaseModel, df: pd.DataFrame):
    """run the model once (no inference) and return the numpyro trace."""
    features, coords, dims = model.prepare_data(df)

    state = ModelAnalysisState(
        model_name=model.name(),
        model_builder=model,
        features=features,
        coords=coords,
        dims=dims,
    )

    # quick fitting for tests
    state.model_config = state.model_builder.config.merge_in_dict(
        config_dict={
            "fit": {"samples": 1, "chains": 1, "warmup": 1, "progress_bar": False},
        }
    )

    fit(state)

    return state


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
    assert features["obs"].shape == (
        2,
    )  # one row per task, model not included in betabinomal defaults.
    assert all(features["total_count"] == jnp.array([2, 2], dtype=jnp.int32))

    # auto-generated indexing helpers
    assert "task_effects_index" in features
    assert "num_task_effects" in features
    assert features["num_task_effects"] == 2

    # coords / dims
    assert "task" in coords and len(coords["task"]) == 2
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
    assert features["obs"].shape == (
        2,
    )  # one row per task - not default to include model in binomial
    assert all(features["total_count"] == jnp.array([2, 2], dtype=jnp.int32))

    # auto-generated indexing helpers
    assert "task_effects_index" in features
    assert "num_task_effects" in features
    assert features["num_task_effects"] == 2

    # coords / dims
    assert "task" in coords and len(coords["task"]) == 2
    assert dims["task_effects"] == ["task"]


def test_betabinomial_model_trace_has_expected_nodes(
    beta_binomial_model: BetaBinomial, sample_df: pd.DataFrame
):
    """
    A single forward pass should create the right sample/deterministic names
    without throwing.
    """
    state = _fit_model(beta_binomial_model, sample_df)

    inference_data = state.inference_data

    # Core parameters
    assert "overall_mean" in inference_data.posterior
    assert "task_effects" in inference_data.posterior
    assert "dispersion_phi" in inference_data.posterior

    # Deterministics + likelihood
    assert "p_bar" in inference_data.posterior
    assert "success_prob" in inference_data.posterior


def test_binomial_model_trace_has_expected_nodes(
    binomial_model: Binomial,
    sample_df: pd.DataFrame,
):
    state = _fit_model(binomial_model, sample_df)
    posterior = state.inference_data.posterior

    assert "overall_mean" in posterior
    assert "task_effects" in posterior
    assert "dispersion_phi" not in posterior
    assert "success_prob" in posterior
