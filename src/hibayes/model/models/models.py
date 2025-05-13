from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.distributions.distribution import DistributionLike

from hibayes.utils import init_logger

logger = init_logger()

Features = Dict[str, float | int | jnp.ndarray | np.ndarray]
Coords = Dict[
    str, List[Any]
]  # arviz information on how to map indexes to names. Here we store both the coords and the dims values
Dims = Dict[str, List[str]]
Method = Literal["NUTS", "HMC"]  # MCMC sampler type
ChainMethod = Literal["parallel", "sequential", "vectorised"]


@dataclass(frozen=True, slots=True)
class FitConfig:
    method: Method = "NUTS"
    samples: int = 1000
    warmup: int = 500
    chains: int = 4
    seed: int = 0
    progress_bar: bool = True
    parallel: bool = True
    chain_method: ChainMethod = "parallel"
    target_accept: float = 0.8
    max_tree_depth: int = 10

    def merged(self, **updates: Any) -> "FitConfig":
        """Return a *new* FitConfig with `updates` applied."""
        return replace(self, **updates)


@dataclass(frozen=True, slots=True)
class PriorConfig:
    distribution: DistributionLike  # The prior distribution
    distribution_args: Dict[
        str, Any
    ] | None = None  # Arguments for the prior distribution

    DISTRIBUTION_MAP: ClassVar[Dict[str, Type[DistributionLike]]] = {
        "normal": dist.Normal,
        "uniform": dist.Uniform,
        "exponential": dist.Exponential,
        "gamma": dist.Gamma,
        "bernoulli": dist.Bernoulli,
        "half_normal": dist.HalfNormal,
        "beta": dist.Beta,
    }

    def get_distribution(self) -> DistributionLike:
        """
        Get the prior distribution with the specified arguments.
        """
        if self.distribution_args is None:
            return self.distribution()
        return self.distribution(**self.distribution_args)

    def merged(self, config_dict: Dict[str, Any]) -> "PriorConfig":
        """
        Merge a dictionary into the existing configuration.
        """
        distribution_name = config_dict.pop("distribution", None)
        distribution_args = config_dict.pop("distribution_args", None)
        old_distribution_args = self.distribution_args or {}

        # if update to distribution
        if (
            distribution_name
            and not self.DISTRIBUTION_MAP[distribution_name] == self.distribution
        ):
            if distribution_name not in self.DISTRIBUTION_MAP:
                raise ValueError(
                    f"Invalid distribution name '{distribution_name}'. Valid options are: {list(self.DISTRIBUTION_MAP.keys())}"
                )
            distribution_class = self.DISTRIBUTION_MAP[distribution_name]
            config_dict["distribution"] = distribution_class

            # as new distribution class we should remove the old args
            old_distribution_args = {}

        if distribution_args is not None:
            if not isinstance(distribution_args, dict):
                raise ValueError(
                    "distribution_args must be a dictionary of arguments for the distribution."
                )
            old_distribution_args.update(distribution_args)
        config_dict["distribution_args"] = old_distribution_args
        return replace(self, **config_dict)


@dataclass(frozen=True, slots=True)
class RandomEffectConfig:
    """Describe one random effect term (optionally with random slopes).
    So far this makes sense to keep separate from ParameterConfig due to
    requring information from data such as num models, num tasks, etc."""

    groups: List[
        str, ...
    ]  # grouping columns defining the levels - this is used in _prepare_data. len(groups) == 1 for random effects, len(groups) > 1 for interaction
    name: str
    prior: Optional[PriorConfig] = None
    slopes: Optional[List[str, ...]] = None  # group specific slopes

    def merged(self, **updates: Any) -> "RandomEffectConfig":
        if "name" in updates and updates["name"] != self.name:
            raise ValueError("Cannot change name for an existing RE config.")
        if "prior" in updates and self.prior is not None:
            updates["prior"] = self.prior.merged(**updates.pop("prior"))  # type: ignore[arg-type]
        return replace(self, **updates)


@dataclass(frozen=True, slots=True)
class ParameterConfig:
    name: str  # Name of the parameter in the model
    prior: PriorConfig | None = None  # Prior distribution for the parameter

    def merge_in_dict(self, config_dict: Dict[str, Any]) -> "ParameterConfig":
        new_param_name = config_dict.get("name")
        if new_param_name != self.name:
            raise ValueError(
                f"Parameter name mismatch: '{new_param_name}' does not match '{self.name}'. You cannot update parameter names as they are called in the model."
            )
        if new_prior := config_dict.get("prior", None):
            if not self.prior:
                raise ValueError(
                    "You cannot update the prior of a parameter that does not have one."
                )
            config_dict["prior"] = self.prior.merged(new_prior)
        return replace(self, **config_dict)


@dataclass
class ModelConfig:
    column_map: Optional[
        Dict[str, str]
    ] = None  # the key for mapping from data columns to any required_column names
    configurable_parameters: Optional[List[ParameterConfig]] = field(
        default_factory=list
    )  # parameters user can adjust
    random_effects: Optional[List[RandomEffectConfig]] = field(
        default_factory=list
    )  ## random effects user can adjust

    fit: FitConfig = field(default_factory=FitConfig)

    main_effect_params: List[
        str
    ] | None = None  # a list of the main effect parameters in the model which you would like to plot. If None then plot all - warning you might have thousands of parameters in the model

    def __post_init__(self):
        # Ensure no duplicate parameter names
        if self.configurable_parameters is not None:
            param_names = {param.name for param in self.configurable_parameters}
            if len(param_names) != len(self.configurable_parameters):
                raise ValueError(
                    "Duplicate parameter names found in model configuration"
                )
            # make sure intercept & slope parameters don't collide
            # very edge case but the error is not nice otherwise
            re_param_names = []
            if self.random_effects is not None:
                for re_cfg in self.random_effects:
                    re_param_names.append(re_cfg.name)
                    if re_cfg.slopes:
                        re_param_names += [f"{re_cfg.name}_{s}" for s in re_cfg.slopes]
                overlap = set(param_names).intersection(re_param_names)
                if overlap:
                    raise ValueError(
                        f"Parameter names reused by random_effects: {', '.join(overlap)}"
                    )

    def parameter_to_numpyro(self, param: str) -> Dict[str, Any]:
        """Get the parameter configuration for a specific parameter."""
        if self.configurable_parameters is None and self.random_effects is None:
            raise ValueError("No parameters defined in model configuration")

        parameter = self.get_param(param)
        if parameter is not None:
            numpyro_args = {"name": parameter.name}
            if parameter.prior is not None:
                numpyro_args["fn"] = parameter.prior.get_distribution()
            return numpyro_args
        raise ValueError(f"Parameter '{param}' not found in model configuration")

    def get_plot_params(self) -> List[str] | None:
        """Get a list of parameters to plot based on the configuration."""
        return self.main_effect_params

    def get_params(self) -> List[str] | None:
        """Get a list of all configurable parameters in the model config."""
        combined_params = (self.configurable_parameters or []) + (
            self.random_effects or []
        )
        if not combined_params:
            raise ValueError("No parameters defined in model configuration")
        return [param.name for param in combined_params]

    def get_param(self, param: str) -> ParameterConfig | None:
        """Get a specific parameter configuration from the configurable parameters."""
        combined_params = (self.configurable_parameters or []) + (
            self.random_effects or []
        )
        if not combined_params:
            raise ValueError("No parameters defined in model configuration")
        for parameter in combined_params:
            if parameter.name == param:
                return parameter
        return None

    def merge_in_dict(self, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Return a config with a merge of default and user specified values."""
        fit_updates = config_dict.get("fit", {})
        new_fit = self.fit.merged(**fit_updates)

        new_column_map = config_dict.get("column_map", self.column_map)

        new_main_effect_params = config_dict.get(
            "main_effect_params", self.main_effect_params
        )
        new_random_effects = self.random_effects
        new_params = self.configurable_parameters

        if "configurable_parameters" in config_dict:
            param_updates = config_dict["configurable_parameters"]
            if isinstance(param_updates, dict):  # only update a single parameter
                param_updates_iter = [param_updates]
            elif isinstance(param_updates, list):  # assuming a list of Mapping
                if not all(isinstance(param, dict) for param in param_updates):
                    raise ValueError(
                        "To update parameters you need to specify either a list of dicts or a single dict where for each parameter at least the parameter name and what you would like to change is detailed."
                    )
                param_updates_iter = param_updates
            else:
                raise ValueError(
                    "To update parameters you need to specify either a list of dicts or a single dict where for each parameter at least the parameter name and what you would like to change is detailed."
                )
            param_map = {param.name: param for param in self.configurable_parameters}
            for param in param_updates_iter:
                name = param.get("name", None)
                if name not in param_map:
                    raise ValueError(
                        f"Parameter name '{param['name']}' not found in model configuration"
                    )
                param_map[name] = param_map[name].merge_in_dict(param)
            new_params = list(param_map.values())

        if "random_effects" in config_dict:
            random_effects_updates = config_dict["random_effects"]
            if isinstance(random_effects_updates, dict):
                random_effects_updates_iter = [random_effects_updates]
            elif isinstance(random_effects_updates, list):
                if not all(isinstance(param, dict) for param in random_effects_updates):
                    raise ValueError(
                        "To update parameters you need to specify either a list of dicts or a single dict where for each parameter at least the parameter name and what you would like to change is detailed."
                    )
                random_effects_updates_iter = random_effects_updates
            else:
                raise ValueError(
                    "To update parameters you need to specify either a list of dicts or a single dict where for each parameter at least the parameter name and what you would like to change is detailed."
                )
            random_effect_map = {param.name: param for param in self.random_effects}
            for param in random_effects_updates_iter:
                name = param.get("name", None)
                if name not in random_effect_map:
                    raise ValueError(
                        f"Parameter name '{param['name']}' not found in model configuration"
                    )
                random_effect_map[name] = random_effect_map[name].merged(**param)
            new_random_effects = list(random_effect_map.values())

        return replace(
            self,
            column_map=new_column_map,
            fit=new_fit,
            main_effect_params=new_main_effect_params,
            configurable_parameters=new_params,
            random_effects=new_random_effects,
        )


class BaseModel(ABC):
    """Base class for methods for statistical modelling and analysis of inspect data logs."""

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], ModelConfig]] = None,
    ) -> None:
        """
        Return the numpyro model callable

        Parameters
        ----------
        config : Optional[Union[Dict[str, Any], ModelConfig]]
            Configuration dictionary or ModelConfig instance to override defaults

        """
        # Merge default config with provided config
        default_config = self.get_default_config()
        if isinstance(config, dict):
            self.config = default_config.merge_in_dict(config)
        elif isinstance(config, ModelConfig):
            self.config = config
        else:
            self.config = default_config

    @classmethod
    @abstractmethod
    def get_default_config(cls) -> ModelConfig:
        return ModelConfig()

    @abstractmethod
    def build_model(self) -> Callable[..., Any]:
        """
        Build the numpyro model

        Returns:
            Callable: A NumPyro model function that must accept at least:
                - obs (optional): Observed data, defaults to None
        """

        def model(obs=None):
            # Numpyro model implementation that uses obs - observational data
            # Why kwarg with none? this is for prior predictive checks!
            pass

        return model

    def prepare_data(self, data: pd.DataFrame) -> Tuple[Features, Coords, Dims]:
        """
        Prepare data for modeling. Wraps the abstract method _prepare_data so
        to enforce that observed variable in features.
        """
        data_copy = (
            data.rename(columns=self.config.column_map)
            if self.config.column_map
            else data.copy()
        )
        features, coords, dims = self._prepare_data(data_copy)

        if "obs" not in features:
            raise ValueError(
                "The model must have an observation feature named 'obs' in the configuration."
            )
        return features, coords, dims

    @abstractmethod
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[Features, Coords, Dims]:
        """
        Abstract method for preparing data. This should be implemented in subclasses.
        """
        pass

    @classmethod
    def name(cls) -> str:
        """
        Get the name of the model.
        """
        return cls.__name__

    def sample_param(self, name: str) -> jnp.ndarray:
        return numpyro.sample(**self.config.parameter_to_numpyro(name))

    def sample_plate(
        self,
        *,
        plate_name: str,
        size: int,
        param_name: str,
        index: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Helper function for a simple plate
        """
        with numpyro.plate(plate_name, size):
            vals = self.sample_param(param_name)

        return vals[index] if index is not None else vals

    def linear_component(self, **features) -> jnp.ndarray:
        """
        Build the linear component of the model. This seems to be a common pattern
        across models. If we are finding many people are overriding this we should
        move to subclass.
        """
        linear = self.sample_param("overall_mean")

        # loop over random effects
        for re_cfg in self.config.random_effects:
            size = features[f"num_{re_cfg.name}"]
            index = features[f"{re_cfg.name}_index"]

            # intercept
            re_int = self.sample_plate(
                plate_name=f"{re_cfg.name}_plate",
                size=size,
                param_name=re_cfg.name,
                index=index,
            )
            linear = linear + re_int

            # slopes if any
            if re_cfg.slopes:
                for pred in re_cfg.slopes:
                    slope_name = f"{re_cfg.name}_{pred}"
                    slope_draw = self.sample_plate(
                        plate_name=f"{slope_name}_plate",
                        size=size,
                        param_name=slope_name,
                        index=index,
                    )
                    linear = linear + slope_draw * features[pred]
        return linear


class BetaBinomial(BaseModel):
    """flexible BetaBinomial GLM with random effects

    The class is generic. The set of random effects is defined in the config. The default config models 2 random effects with no interaction.
    """

    @classmethod
    def get_default_config(cls) -> ModelConfig:
        re_1 = RandomEffectConfig(
            groups=["model"],
            name="model_effects",
            prior=PriorConfig(
                distribution=dist.Normal,
                distribution_args={"loc": 0.0, "scale": 0.3},
            ),
        )
        re_2 = RandomEffectConfig(
            groups=["task"],
            name="task_effects",
            prior=PriorConfig(
                distribution=dist.Normal,
                distribution_args={"loc": 0.0, "scale": 0.3},
            ),
        )

        params = [
            ParameterConfig(
                name="overall_mean",
                prior=PriorConfig(
                    distribution=dist.Normal,
                    distribution_args={"loc": 0.0, "scale": 0.3},
                ),
            ),
            ParameterConfig(
                name="dispersion_phi",
                prior=PriorConfig(
                    distribution=dist.Gamma,
                    distribution_args={"concentration": 1.0, "rate": 0.1},
                ),
            ),
        ]

        return ModelConfig(
            configurable_parameters=params,
            random_effects=[re_1, re_2],
            main_effect_params=["overall_mean", "model_effects", "success_prob"],
        )

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[Features, Coords, Dims]:
        if self.config.random_effects is None:
            raise ValueError("No random effects defined in the model configuration.")

        # TODO: Magda to check this assumption.
        # collect all grouping columns that appear in any RE
        # each RE should contain only one value per linear predictor.
        random_effect_cols = {g for re in self.config.random_effects for g in re.groups}

        # for values for slopes should also not vary within groups
        needed_predictors = {
            s for re in self.config.random_effects if re.slopes for s in re.slopes
        }

        random_effect_cols |= needed_predictors

        if "n_correct" not in data.columns:
            group_cols = sorted(random_effect_cols)
            data = (
                data.groupby(group_cols, observed=True)
                .agg(
                    n_correct=("score", "sum"),
                    n_total=("score", "count"),
                )
                .reset_index()
            )

        features: Features = {
            "total_count": jnp.asarray(data["n_total"].values, dtype=jnp.int32),
            "obs": jnp.asarray(
                data["n_correct"].values / data["n_total"].values, dtype=jnp.float32
            ),
        }
        coords: Coords = {}
        dims: Dims = {}

        for col in needed_predictors:
            # for now only support continuous predictors. TODO add support for categorical and
            # ordered categorical
            if pd.api.types.is_categorical_dtype(data[col]):
                raise NotImplementedError(
                    f"Categorical predictors are not supported yet. Please convert {col} to a continuous type or define your own custom model."
                )
            features[col] = jnp.asarray(data[col].values, dtype=jnp.float32)

        for re_cfg in self.config.random_effects:
            key_name = "_x_".join(re_cfg.groups)  # note just the name if only one group
            codes, cats = pd.factorize(
                tuple(zip(*(data[g] for g in re_cfg.groups))), sort=True
            )  # we can assume our groups are categorical.
            # should we unpack if only one in group to get rid of ugly tuple?

            idx_name = f"{re_cfg.name}_index"
            size_name = f"num_{re_cfg.name}"

            features[idx_name] = jnp.asarray(codes, dtype=jnp.int32)
            features[size_name] = len(cats)

            coords[key_name] = cats.tolist()
            dims[re_cfg.name] = [key_name]
            if re_cfg.slopes:
                for s in re_cfg.slopes:
                    dims[f"{re_cfg.name}_{s}"] = [key_name]

        return features, coords, dims

    def build_model(self):
        def model(**features):
            linear = self.linear_component(**features)
            # likelihood
            p_bar = numpyro.deterministic("p_bar", jax.nn.sigmoid(linear))

            phi = self.sample_param("dispersion_phi")
            alpha = p_bar * phi
            beta = (1.0 - p_bar) * phi

            success_prob = numpyro.sample("success_prob", dist.Beta(alpha, beta))

            n_tot = features["total_count"]
            if features["obs"] is not None:
                obs_cnt = jnp.round(features["obs"] * n_tot).astype(jnp.int32)

                numpyro.sample(
                    "n_correct",
                    dist.Binomial(n_tot, success_prob),
                    obs=obs_cnt,
                )
            else:
                count_pred = numpyro.sample(
                    "n_correct",
                    dist.Binomial(n_tot, success_prob),
                )
                # we like to plot proportions
                numpyro.deterministic("obs", count_pred / n_tot)

        return model


class Binomial(BetaBinomial):
    @classmethod
    def get_default_config(cls) -> ModelConfig:
        re_1 = RandomEffectConfig(
            groups=["model"],
            name="model_effects",
            prior=PriorConfig(
                distribution=dist.Normal,
                distribution_args={"loc": 0.0, "scale": 0.3},
            ),
        )
        re_2 = RandomEffectConfig(
            groups=["task"],
            name="task_effects",
            prior=PriorConfig(
                distribution=dist.Normal,
                distribution_args={"loc": 0.0, "scale": 0.3},
            ),
        )

        params = [
            ParameterConfig(
                name="overall_mean",
                prior=PriorConfig(
                    distribution=dist.Normal,
                    distribution_args={"loc": 0.0, "scale": 0.3},
                ),
            ),
        ]

        return ModelConfig(
            configurable_parameters=params,
            random_effects=[re_1, re_2],
            main_effect_params=["overall_mean", "model_effects", "success_prob"],
        )

    # same prep data as BetaBinomial

    def build_model(self):
        def model(**features):
            linear = self.linear_component(**features)

            # likelihood
            success_prob = numpyro.deterministic("success_prob", jax.nn.sigmoid(linear))

            n_tot = features["total_count"]
            if features["obs"] is not None:
                obs_cnt = jnp.round(features["obs"] * n_tot).astype(jnp.int32)

                numpyro.sample(
                    "n_correct",
                    dist.Binomial(n_tot, success_prob),
                    obs=obs_cnt,
                )
            else:
                count_pred = numpyro.sample(
                    "n_correct",
                    dist.Binomial(n_tot, success_prob),
                )
                # we like to plot proportions
                numpyro.deterministic("obs", count_pred / n_tot)

        return model
