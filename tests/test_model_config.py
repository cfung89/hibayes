from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import patch

import jax.numpy as jnp
import numpyro.distributions as dist
import pandas as pd
import pytest

from hibayes.model.model import ModelsToRunConfig
from hibayes.model.models import (
    BetaBinomial,
    Binomial,
    FitConfig,
    ModelConfig,
    ParameterConfig,
    PriorConfig,
)


def test_fitconfig_merged():
    cfg = FitConfig()
    new_cfg = cfg.merged(samples=2_000, method="HMC", chains=2)

    assert cfg.samples == 4_000  # original unchanged
    assert new_cfg.samples == 2_000
    assert new_cfg.method == "HMC"
    assert new_cfg.chains == 2


def test_priorconfig():
    pc = PriorConfig(
        distribution=dist.Normal, distribution_args={"loc": 0.0, "scale": 1.0}
    )

    assert pc.distribution is dist.Normal
    assert pc.distribution_args == {"loc": 0.0, "scale": 1.0}


def test_priorconfig_frondict():
    pc = PriorConfig.from_dict(
        {
            "distribution": "uniform",
            "distribution_args": {"low": 0.0, "high": 1.0},
        }
    )

    assert pc.distribution is dist.Uniform
    assert pc.distribution_args == {"low": 0.0, "high": 1.0}


def test_parameterconfig_name_mismatch_raises():
    param = ParameterConfig(name="alpha", prior=PriorConfig(distribution=dist.Normal))
    with pytest.raises(ValueError):
        param.merge_in_dict({"name": "beta"})


def test_modelconfig_merge_updates(default_model_cfg: ModelConfig):
    updates = {
        "fit": {"samples": 500},
        "configurable_parameters": {
            "name": "overall_mean",
            "prior": {
                "distribution": "uniform",
                "distribution_args": {"low": 0.0, "high": 1.0},
            },
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
    assert default_model_cfg.fit.samples == 4_000


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


def test_modeltorun_from_dict_with_configs():
    cfg = ModelsToRunConfig.from_dict(
        {"models": {"name": "BetaBinomial", "config": {"fit": {"samples": 50}}}}
    )

    (model_cls, model_cfg), *_ = cfg.enabled_models
    assert model_cls is BetaBinomial
    assert model_cfg == {"fit": {"samples": 50}}


def test_modelstorun_from_dict_with_configs():
    cfg = ModelsToRunConfig.from_dict(
        {
            "models": [
                {"name": "BetaBinomial", "config": {"fit": {"samples": 50}}},
                "Binomial",
            ]
        }
    )
    model_cls, model_cfg = cfg.enabled_models[0]
    assert model_cls is BetaBinomial
    assert model_cfg == {"fit": {"samples": 50}}

    model_cls, model_cfg = cfg.enabled_models[1]
    assert model_cls is Binomial
    assert model_cfg is None


def test_modelstorun_invalid_name_fallback_to_defaults():
    with patch.object(ModelsToRunConfig, "DEFAULT_MODELS", ["BetaBinomial"]):
        cfg = ModelsToRunConfig.from_dict({"models": ["NotARealModel"]})

    enabled_names = [cls.__name__ for cls, _ in cfg.enabled_models]
    assert enabled_names == ["BetaBinomial"], "Should fall back to default model list"


def test_build_models_produces_analysis_state(sample_df: pd.DataFrame):
    cfg = ModelsToRunConfig()  # default config
    states = list(cfg.build_models(sample_df))

    assert len(states) == 1

    state = states[0]
    # Basic sanity checks
    assert state.model_name == "BetaBinomial"
    assert "obs" in state.features
    assert len(state.features["obs"]) == 2  # 2 tasks with 2 scores each

    cfg2 = ModelsToRunConfig().from_dict(
        {
            "models": {
                "name": "BetaBinomial",
                "config": {
                    "random_effects": {
                        "name": "model_effects",
                        "groups": ["model"],
                        "prior": {
                            "distribution": "uniform",
                            "distribution_args": {"low": 0.0, "high": 0.3},
                        },
                    }
                },
            }
        }
    )

    states2 = list(cfg2.build_models(sample_df))
    state2 = states2[0]
    assert state2.model_name == "BetaBinomial"

    assert "model" in state2.coords and len(state2.coords["model"]) == 2  # m1, m2
    assert "model_effects_index" in state2.features
    assert len(state2.features["obs"]) == 4  # 2 models with 2 tasks each with 1 score


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
    bb_multi_re_model_2: BetaBinomial,
    sample_df_2: pd.DataFrame,
):
    """
    Verify that prepare_data:
     - aggregates into n_correct / n_total per (model,task),
     - computes obs = n_correct / n_total,
     - carries difficulty through,
     - builds indexes, counts, coords & dims for all REs.
    """
    features, coords, dims = bb_multi_re_model_2.prepare_data(sample_df_2)

    # we have exactly 4 groups => 4 obs
    assert features["obs"].shape == (4,)
    # check fixture
    assert jnp.allclose(
        features["obs"],
        jnp.asarray([0.8, 0.5, 0.7, 0.6], dtype=jnp.float32),
    )

    # n_correct and n_total should have been created
    assert jnp.all(features["n_correct"] == jnp.asarray([4, 1, 7, 3]))
    assert jnp.all(features["total_count"] == jnp.asarray([5, 2, 10, 5]))

    # check RE index/count fields
    re_names: List[str] = [
        "model_effects",
        "task_effects",
        "difficulty_effects",
        "model_task_int",
    ]
    for name in re_names:
        assert f"{name}_index" in features
        assert f"num_{name}" in features
        # every dim label references a coord key
        assert dims[name][0] in coords

    # coords for each grouping variable
    assert coords["model"] == ["m1", "m2"]
    assert coords["task"] == ["t1", "t2"]
    # interaction coord order = unique (model,task) pairs
    assert coords["model_x_task"] == [
        ("m1_x_t1"),
        ("m1_x_t2"),
        ("m2_x_t1"),
        ("m2_x_t2"),
    ]

    # counts of levels for each RE
    assert features["num_model_effects"] == 2
    assert features["num_task_effects"] == 2
    assert features["num_model_task_int"] == 4
