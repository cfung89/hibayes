from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import jax.numpy as jnp
import numpyro.distributions as dist
import pandas as pd
import pytest

from hibayes.model.model import ModelsToRunConfig
from hibayes.model.models import (
    BetaBinomial,
    FitConfig,
    ModelConfig,
    ParameterConfig,
    PriorConfig,
    RandomEffectConfig,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A minimal dataframe suitable for the default BetaBinomial model."""
    return pd.DataFrame(
        {
            "model": ["m1", "m1", "m2", "m2"],
            "task": ["t1", "t1", "t1", "t1"],
            "score": [1, 0, 1, 0],
        }
    )


@pytest.fixture
def default_model_cfg() -> ModelConfig:
    """The out-of-the-box config for BetaBinomial."""
    return BetaBinomial.get_default_config()


@pytest.fixture
def agg_df() -> pd.DataFrame:
    """
    Already-aggregated data (so n_correct / n_total present) to avoid the
    internal grouping step that would otherwise drop the 'difficulty' column.
    """
    return pd.DataFrame(
        {
            "model": ["m1", "m1", "m2", "m2"],
            "task": ["t1", "t2", "t1", "t2"],
            "difficulty": [0.20, 0.60, 0.30, 0.70],
            "n_correct": [8, 7, 5, 6],
            "n_total": [10, 10, 10, 10],
        }
    )


@pytest.fixture
def bb_multi_re_model() -> BetaBinomial:
    """
    BetaBinomial whose config contains

    * model level random intercept and random slope (difficulty)
    * task level random intercept
    * model×task interaction intercept
    """
    base_cfg = BetaBinomial.get_default_config()

    # add a random sope
    re_model_slope = replace(base_cfg.random_effects[0], slopes=["difficulty"])

    # no change
    re_task = base_cfg.random_effects[1]

    # interaction RE over (model, task)
    re_inter = RandomEffectConfig(
        groups=["model", "task"],
        name="model_task_int",
        prior=PriorConfig(
            distribution=dist.Normal, distribution_args={"loc": 0.0, "scale": 0.3}
        ),
    )

    cfg = ModelConfig(
        configurable_parameters=base_cfg.configurable_parameters,
        random_effects=[re_model_slope, re_task, re_inter],
        main_effect_params=base_cfg.main_effect_params,
        fit=base_cfg.fit,
    )

    return BetaBinomial(config=cfg)


def test_fitconfig_merged():
    cfg = FitConfig()
    new_cfg = cfg.merged(samples=2_000, method="HMC", chains=2)

    assert cfg.samples == 1_000  # original unchanged
    assert new_cfg.samples == 2_000
    assert new_cfg.method == "HMC"
    assert new_cfg.chains == 2


def test_priorconfig_merge_change_distribution():
    pc = PriorConfig(
        distribution=dist.Normal, distribution_args={"loc": 0.0, "scale": 1.0}
    )

    pc2 = pc.merged(
        {
            "distribution": "uniform",
            "distribution_args": {"low": 0.0, "high": 1.0},
        }
    )

    assert pc2.distribution is dist.Uniform
    assert pc2.distribution_args == {"low": 0.0, "high": 1.0}

    # Original should be untouched
    assert pc.distribution is dist.Normal


def test_parameterconfig_name_mismatch_raises():
    param = ParameterConfig(name="alpha")
    with pytest.raises(ValueError):
        param.merge_in_dict({"name": "beta"})


# helpers
def _uniform_prior_dict() -> Dict[str, Any]:
    return {
        "distribution": "uniform",
        "distribution_args": {"low": 0.0, "high": 1.0},
    }


def test_modelconfig_merge_updates(default_model_cfg: ModelConfig):
    updates = {
        "fit": {"samples": 500},
        "configurable_parameters": {
            "name": "overall_mean",
            "prior": _uniform_prior_dict(),
        },
        "main_effect_params": ["overall_mean"],
    }

    merged = default_model_cfg.merge_in_dict(updates)

    # Fit updates
    assert merged.fit.samples == 500

    # Parameter updates
    param = merged.get_param("overall_mean")
    assert param is not None and param.prior is not None
    assert param.prior.distribution is dist.Uniform
    assert param.prior.distribution_args == {"low": 0.0, "high": 1.0}

    # List-valued updates
    assert merged.main_effect_params == ["overall_mean"]

    # Original untouched
    assert default_model_cfg.fit.samples == 1_000


def test_modelconfig_merge_unknown_param_raises(default_model_cfg: ModelConfig):
    with pytest.raises(ValueError):
        default_model_cfg.merge_in_dict(
            {"configurable_parameters": {"name": "does_not_exist"}}
        )


def test_modelstorun_default_models():
    cfg = ModelsToRunConfig()
    enabled_classes = [cls for cls, _ in cfg.enabled_models]

    assert len(enabled_classes) == 1
    assert BetaBinomial in enabled_classes


def test_modelstorun_from_dict_list():
    cfg = ModelsToRunConfig.from_dict({"models": ["BetaBinomial", "Binomial"]})
    enabled_names = {cls.__name__ for cls, _ in cfg.enabled_models}

    assert enabled_names == {"BetaBinomial", "Binomial"}


def test_modelstorun_from_dict_with_configs():
    cfg = ModelsToRunConfig.from_dict(
        {"models": {"BetaBinomial": {"fit": {"samples": 50}}}}
    )

    (model_cls, model_cfg), *_ = cfg.enabled_models
    assert model_cls is BetaBinomial
    assert model_cfg == {"fit": {"samples": 50}}


def test_modelstorun_invalid_name_fallback_to_defaults():
    with patch.object(ModelsToRunConfig, "DEFAULT_MODELS", ["BetaBinomial"]):
        cfg = ModelsToRunConfig.from_dict({"models": ["NotARealModel"]})

    enabled_names = [cls.__name__ for cls, _ in cfg.enabled_models]
    assert enabled_names == ["BetaBinomial"], "Should fall back to default model list"


def test_build_models_produces_analysis_state(sample_df: pd.DataFrame):
    cfg = ModelsToRunConfig()
    states = list(cfg.build_models(sample_df))

    assert len(states) == 1

    state = states[0]
    # Basic sanity checks
    assert state.model_name == "BetaBinomial"
    assert "obs" in state.features
    assert len(state.features["obs"]) == 2  # 2 tasks with 2 scores each
    assert "model" in state.coords and len(state.coords["model"]) == 2  # m1, m2


def test_modelstorun_from_none_returns_default():
    cfg = ModelsToRunConfig.from_dict(None)
    assert len(cfg.enabled_models) == 1
    assert cfg.enabled_models[0][0] is BetaBinomial


def test_build_models_with_custom_class_and_default_args(tmp_path: Path):
    """
    Load a user model via custom_models (no explicit args),
    then ensure build_models yields the expected AnalysisState.
    """
    file = tmp_path / "user_models.py"
    file.write_text(
        """
from hibayes.model.models import BaseModel, ModelConfig
import pandas as pd
class PipelineDummy(BaseModel):
    @classmethod
    def get_default_config(cls): return ModelConfig()
    def _prepare_data(self, data: pd.DataFrame): return {"obs": []}, {"coords": {}}, {"dims": {}}
    def build_model(self): return lambda obs=None: None
"""
    )

    cfg = ModelsToRunConfig.from_dict(
        {"custom_models": {"path": str(file), "classes": ["PipelineDummy"]}}
    )

    assert cfg.enabled_models[0][1] is None  # default args
    state = next(cfg.build_models(pd.DataFrame()))
    assert state.model_name == "PipelineDummy" and "obs" in state.features


def test_custom_model_with_explicit_args(tmp_path: Path):
    """
    Same as above but now pass a per model config dict and verify it
    reaches the model instance.
    """
    file = tmp_path / "user_models.py"
    file.write_text(
        """
from hibayes.model.models import BaseModel, ModelConfig
import pandas as pd
class ArgsModel(BaseModel):
    @classmethod
    def get_default_config(cls): return ModelConfig()
    def _prepare_data(self, data: pd.DataFrame): return {"obs": []}, {"coords": {}}, {"dims": {}}
    def build_model(self): return lambda obs=None: None
"""
    )

    model_cfg = {"fit": {"samples": 42}}
    cfg = ModelsToRunConfig.from_dict(
        {
            "custom_models": {
                "path": str(file),
                "classes": {"ArgsModel": model_cfg},
            }
        }
    )

    mdl_cls, cfg_dict = cfg.enabled_models[0]
    assert mdl_cls.__name__ == "ArgsModel" and cfg_dict == model_cfg

    # The config dict should be applied when the model is instantiated
    state = next(cfg.build_models(pd.DataFrame()))
    assert state.model_builder.config.fit.samples == 42


def test_prepare_data_multiple_random_effects(
    bb_multi_re_model: BetaBinomial,
    agg_df: pd.DataFrame,
):
    """
    Verify that _index / num_ helpers, coords and dims are created for
    all REs and that the slope introduces an extra dim.
    """
    features, coords, dims = bb_multi_re_model.prepare_data(agg_df)

    assert features["obs"].shape == (4,)
    assert jnp.allclose(
        features["obs"], jnp.asarray([0.8, 0.7, 0.5, 0.6], dtype=jnp.float32)
    )
    assert "difficulty" in features and features["difficulty"].shape == (4,)

    re_names: List[str] = ["model_effects", "task_effects", "model_task_int"]
    for name in re_names:
        assert f"{name}_index" in features
        assert f"num_{name}" in features
        assert dims[name][0] in coords  # every dim label references a coord key

    # slopes
    assert "model_effects_difficulty" in dims
    assert dims["model_effects_difficulty"] == ["model"]

    # coords
    assert coords["model"] == [
        ("m1",),
        ("m2",),
    ]  # should we consider unpacking if one one var in group?
    assert coords["task"] == [("t1",), ("t2",)]
    # interaction coord order should mirror the unique (model, task) pairs
    assert coords["model_x_task"] == [
        ("m1", "t1"),
        ("m1", "t2"),
        ("m2", "t1"),
        ("m2", "t2"),
    ]

    # counts
    assert features["num_model_effects"] == 2
    assert features["num_task_effects"] == 2
    assert features["num_model_task_int"] == 4
