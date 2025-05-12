from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import numpyro.distributions as dist
import pandas as pd
import pytest

from hibayes.model.model import ModelsToRunConfig
from hibayes.model.models import (
    FitConfig,
    ModelBetaBinomial,
    ModelConfig,
    ParameterConfig,
    PriorConfig,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A minimal dataframe suitable for the default ModelBetaBinomial."""

    return pd.DataFrame(
        {
            "model": ["m1", "m1", "m2", "m2"],
            "task": ["t1", "t1", "t1", "t1"],
            "score": [1, 0, 1, 0],
        }
    )


@pytest.fixture
def default_model_cfg() -> ModelConfig:
    """The out of the box config for ModelBetaBinomial."""

    return ModelBetaBinomial.get_default_config()


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

    # List valued updates
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
    assert ModelBetaBinomial in enabled_classes


def test_modelstorun_from_dict_list():
    cfg = ModelsToRunConfig.from_dict(
        {"models": ["ModelBetaBinomial", "ModelBinomial"]}
    )
    enabled_names = {cls.__name__ for cls, _ in cfg.enabled_models}

    assert enabled_names == {"ModelBetaBinomial", "ModelBinomial"}


def test_modelstorun_from_dict_with_configs():
    cfg = ModelsToRunConfig.from_dict(
        {"models": {"ModelBetaBinomial": {"fit": {"samples": 50}}}}
    )

    (model_cls, model_cfg), *_ = cfg.enabled_models
    assert model_cls is ModelBetaBinomial
    assert model_cfg == {"fit": {"samples": 50}}


def test_modelstorun_invalid_name_fallback_to_defaults():
    with patch.object(ModelsToRunConfig, "DEFAULT_MODELS", ["ModelBetaBinomial"]):
        cfg = ModelsToRunConfig.from_dict({"models": ["NotARealModel"]})

    enabled_names = [cls.__name__ for cls, _ in cfg.enabled_models]
    assert enabled_names == [
        "ModelBetaBinomial"
    ], "Should fall back to default model list"


def test_build_models_produces_analysis_state(sample_df: pd.DataFrame):
    cfg = ModelsToRunConfig()

    states = list(cfg.build_models(sample_df))

    assert len(states) == 1

    state = states[0]
    # Basic sanity checks
    assert state.model_name == "ModelBetaBinomial"
    assert "obs" in state.features
    assert len(state.features["obs"]) == 2  # 2 tasks with 2 scores each
    assert "model" in state.coords["coords"]
    assert len(state.coords["coords"]["model"]) == 2  # m1, m2


def test_modelstorun_from_none_returns_default():
    cfg = ModelsToRunConfig.from_dict(None)
    assert len(cfg.enabled_models) == 1
    assert cfg.enabled_models[0][0] is ModelBetaBinomial


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
    def _prepare_data(self, data: pd.DataFrame): return {"obs": []}, {"coords": {}, "dims": {}}
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
    def _prepare_data(self, data: pd.DataFrame): return {"obs": []}, {"coords": {}, "dims": {}}
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
