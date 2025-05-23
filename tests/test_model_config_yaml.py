from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import jax.numpy as jnp
import numpyro.distributions as dist
import pandas as pd
import pytest
import yaml

from hibayes.model.model import ModelsToRunConfig
from hibayes.model.models import PriorConfig


def write_yaml(
    tmp_path: Path, content: Dict[str, Any], name: str = "config.yaml"
) -> Path:
    """
    Helper to write a YAML dict to a temporary file and return its path.
    """
    file = tmp_path / name
    file.write_text(yaml.safe_dump(content))
    return file


def test_interaction_yaml(
    cfg_dir: Path,
    sample_df_2: pd.DataFrame,
):
    cfg = ModelsToRunConfig.from_yaml(str(cfg_dir / "interaction.yaml"))
    state = next(cfg.build_models(sample_df_2))

    assert state.dims["task_effects"] == ["task"]
    assert state.dims["task_model_effects"] == ["model_x_task"]
    assert "model_x_task" in state.coords
    # indexing helpers exist
    assert (
        "task_effects_index" in state.features and "num_task_effects" in state.features
    )


def test_delete_yaml(
    cfg_dir: Path,
    sample_df: pd.DataFrame,
):
    cfg = ModelsToRunConfig.from_yaml(str(cfg_dir / "delete_re.yaml"))
    state = next(cfg.build_models(sample_df))
    # check that the model is not in the config

    assert "task_effects" not in state.dims
    assert "model_effects" in state.dims


def test_change_prior_yaml(
    cfg_dir: Path,
    sample_df: pd.DataFrame,
):
    cfg = ModelsToRunConfig.from_yaml(str(cfg_dir / "change_prior.yaml"))
    state = next(cfg.build_models(sample_df))
    # check that the model is not in the config
    param = state.model_config.get_param("overall_mean")

    assert param.prior.distribution is dist.Normal
    assert param.prior.distribution_args == {
        "loc": 0.0,
        "scale": 10.0,
    }


def test_two_models_of_same_class(
    cfg_dir: Path,
    sample_df: pd.DataFrame,
):
    cfg = ModelsToRunConfig.from_yaml(str(cfg_dir / "two_models_same_class.yaml"))
    states = list(cfg.build_models(sample_df))
    assert len(states) == 2
    assert states[0].model_name == "Binomial_version_1"
    assert states[1].model_name == "Binomial_version_2"
    assert states[0].model_config.tag == "version_1"
    assert states[1].model_config.tag == "version_2"

    assert states[0].model_config.random_effects[0].prior.distribution_args == {
        "loc": 0.0,
        "scale": 0.3,
    }
    assert states[1].model_config.random_effects[0].prior.distribution_args == {
        "loc": 0.0,
        "scale": 10.0,
    }


@pytest.mark.parametrize(
    "yaml_content, expected_models",
    [
        # single model by name
        ({"models": "Binomial"}, ["Binomial"]),
        # single model with config
        (
            {"models": {"name": "BetaBinomial", "config": {"fit": {"samples": 10}}}},
            ["BetaBinomial"],
        ),
        # list of models, mixed styles
        (
            {
                "models": [
                    "Binomial",
                    {"name": "BetaBinomial", "config": {"fit": {"chains": 2}}},
                ]
            },
            ["Binomial", "BetaBinomial"],
        ),
    ],
)
def test_models_from_yaml(
    tmp_path: Path, yaml_content: Dict[str, Any], expected_models: list[str]
):
    cfg_file = write_yaml(tmp_path, {"models": yaml_content["models"]})
    cfg = ModelsToRunConfig.from_yaml(str(cfg_file))
    names = [cls.__name__ for cls, cfg_dict in cfg.enabled_models]
    assert names == expected_models


@pytest.mark.parametrize(
    "invalid_yaml",
    [
        # missing name in dict
        {"models": {"config": {}}},
        # unknown model
        {"models": "NonExistentModel"},
        # incomplete list entry
        {"models": [{"config": {}}]},
    ],
)
def test_invalid_model_yaml(tmp_path: Path, invalid_yaml: Dict[str, Any]):
    cfg_file = write_yaml(tmp_path, invalid_yaml)
    # fallback to default when unknown
    cfg = ModelsToRunConfig.from_yaml(str(cfg_file))
    # default should be BetaBinomial
    names = [cls.__name__ for cls, _ in cfg.enabled_models]
    assert names == [ModelsToRunConfig.DEFAULT_MODELS[0]]


@pytest.mark.parametrize(
    "interaction_groups",
    [
        ["task", "model"],
        ["model", "task"],
    ],
)
def test_interaction_group_order_equivalence(
    tmp_path: Path, sample_df: pd.DataFrame, interaction_groups: list[str]
):
    # YAML with interaction random effect in different order
    content = {
        "models": {
            "name": "Binomial",
            "config": {
                "random_effects": [
                    {
                        "name": "rtm",
                        "groups": interaction_groups,
                        "prior": {
                            "distribution": "normal",
                            "distribution_args": {"loc": 0, "scale": 1},
                        },
                    }
                ]
            },
        }
    }
    cfg_file = write_yaml(tmp_path, content)
    cfg = ModelsToRunConfig.from_yaml(str(cfg_file))
    state = next(cfg.build_models(sample_df))
    # dims key should use joined order alphabetical
    key = "model_x_task"
    assert key in state.coords
    assert state.dims["rtm"] == [key]


def test_custom_prior_settings(tmp_path: Path):
    # Test yaml mapping to PriorConfig
    content = {
        "models": {
            "name": "Binomial",
            "config": {
                "random_effects": [
                    {
                        "name": "task_effects",
                        "groups": ["task"],
                        "prior": {
                            "distribution": "uniform",
                            "distribution_args": {"low": 0.1, "high": 0.9},
                        },
                    }
                ]
            },
        }
    }
    cfg_file = write_yaml(tmp_path, content)
    cfg = ModelsToRunConfig.from_yaml(str(cfg_file))
    model_cls, model_cfg = cfg.enabled_models[0]
    # instantiate and inspect config object
    model_instance = model_cls(config=model_cfg)
    re_cfgs = model_instance.config.random_effects
    assert len(re_cfgs) == 1
    prior = re_cfgs[0].prior
    assert isinstance(prior, PriorConfig)
    assert prior.distribution is dist.Uniform
    assert prior.distribution_args == {"low": 0.1, "high": 0.9}


def test_build_model_callable_and_prepare(sample_df, tmp_path: Path):
    # ensure that building and preparing data works after yaml load
    content = {"models": "BetaBinomial"}
    cfg_file = write_yaml(tmp_path, content)
    cfg = ModelsToRunConfig.from_yaml(str(cfg_file))
    state = next(cfg.build_models(sample_df))
    # build_model should be callable
    model_fn = state.model_builder.build_model()
    assert callable(model_fn)
    # prepare_data should include obs
    features, coords, dims = state.model_builder.prepare_data(sample_df)
    assert "obs" in features
    assert isinstance(features["obs"], jnp.ndarray)
