from __future__ import annotations

from pathlib import Path

import numpyro.distributions as dist
import pandas as pd
import pytest

from hibayes.model.models import (
    BetaBinomial,
    Binomial,
    ModelConfig,
    PriorConfig,
    RandomEffectConfig,
)


@pytest.fixture
def default_model_cfg() -> ModelConfig:
    """The out-of-the-box config for BetaBinomial."""
    return BetaBinomial.get_default_config()


@pytest.fixture
def bb_multi_re_model() -> BetaBinomial:
    """
    BetaBinomial whose config contains

    * model level random intercept
    * task level random intercept
    * model×task interaction intercept
    """
    base_cfg = BetaBinomial.get_default_config()

    # add a random effect
    re_model_effect = RandomEffectConfig(
        name="model_effects",
        groups=["model"],
        prior=PriorConfig(
            distribution=dist.Uniform,
            distribution_args={"low": 0.0, "high": 0.3},
        ),
    )
    # no change
    re_task = base_cfg.random_effects[0]

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
        random_effects=[re_model_effect, re_task, re_inter],
        main_effect_params=base_cfg.main_effect_params,
        fit=base_cfg.fit,
    )

    return BetaBinomial(config=cfg)


@pytest.fixture
def bb_multi_re_model_2() -> BetaBinomial:
    """
    BetaBinomial whose config contains

    * model level random intercept
    * task level random intercept
    * difficulty interaction intercept
    * model×task interaction intercept
    """
    base_cfg = BetaBinomial.get_default_config()

    # add a random effect
    re_model_effect = RandomEffectConfig(
        name="model_effects",
        groups=["model"],
        prior=PriorConfig(
            distribution=dist.Uniform,
            distribution_args={"low": 0.0, "high": 0.3},
        ),
    )
    # no change
    re_task = base_cfg.random_effects[0]

    # add difficulty as effect
    re_difficulty = RandomEffectConfig(
        name="difficulty_effects",
        groups=["difficulty"],
        prior=PriorConfig(
            distribution=dist.Normal, distribution_args={"loc": 0.0, "scale": 0.3}
        ),
    )

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
        random_effects=[re_model_effect, re_task, re_difficulty, re_inter],
        main_effect_params=base_cfg.main_effect_params,
        fit=base_cfg.fit,
    )

    return BetaBinomial(config=cfg)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A minimal dataframe suitable for the default BetaBinomial model."""
    return pd.DataFrame(
        {
            "model": ["m1", "m1", "m2", "m2"],
            "task": ["t1", "t2", "t1", "t2"],
            "score": [1, 0, 1, 0],
        }
    )


@pytest.fixture
def sample_df_2() -> pd.DataFrame:
    """
    A dataframe with multiple observations per (model, task) so that
    BetaBinomial.prepare_data can compute n_correct and n_total.
    """
    rows = []
    specs = [
        ("m1", "t1", 0.20, [1, 1, 1, 1, 0]),  # 4/5 = 0.8
        ("m1", "t2", 0.60, [1] * 7 + [0] * 3),  # 7/10 = 0.7
        ("m2", "t1", 0.30, [1, 0]),  # 1/2 = 0.5
        ("m2", "t2", 0.70, [1, 1, 1] + [0, 0]),  # 3/5 = 0.6
    ]
    for model, task, diff, scores in specs:
        for score in scores:
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "difficulty": diff,
                    "score": score,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cfg_dir() -> Path:
    """tests/model_configs directory."""
    return Path(__file__).parent / "model_configs"


@pytest.fixture
def beta_binomial_model() -> BetaBinomial:
    return BetaBinomial()


@pytest.fixture
def binomial_model() -> Binomial:
    return Binomial()
