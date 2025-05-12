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
    str, Dict[str, jnp.ndarray | np.ndarray]
]  # arviz information on how to map indexes to names. Here we store both the coords and the dims values
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

    def parameter_to_numpyro(self, param: str) -> Dict[str, Any]:
        """
        Get the parameter configuration for a specific parameter.
        """
        if self.configurable_parameters is None:
            raise ValueError("No parameters defined in model configuration")

        parameter = self.get_param(param)
        if parameter is not None:
            numpyro_args = {"name": parameter.name}
            if parameter.prior is not None:
                numpyro_args["fn"] = parameter.prior.get_distribution()
            return numpyro_args
        raise ValueError(f"Parameter '{param}' not found in model configuration")

    def get_plot_params(self) -> List[str] | None:
        """
        Get a list of parameters to plot based on the configuration."""
        return self.main_effect_params

    def get_params(self) -> List[str] | None:
        """
        Get a list of all configurable parameters in the model config."""
        if self.configurable_parameters is None:
            return None
        return [param.name for param in self.configurable_parameters]

    def get_param(self, param: str) -> ParameterConfig | None:
        """
        Get a specific parameter configuration from the configurable parameters.
        """
        if self.configurable_parameters is None:
            raise ValueError("No parameters defined in model configuration")
        for parameter in self.configurable_parameters:
            if parameter.name == param:
                return parameter
        return None

    def merge_in_dict(self, config_dict: Dict[str, Any]) -> "ModelConfig":
        """
        Return a config with a merge of default and user specified values.
        """

        fit_updates = config_dict.get("fit", {})
        new_fit = self.fit.merged(**fit_updates)

        new_column_map = config_dict.get("column_map", self.column_map)

        new_main_effect_params = config_dict.get(
            "main_effect_params", self.main_effect_params
        )

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
        else:
            new_params = self.configurable_parameters

        return replace(
            self,
            column_map=new_column_map,
            fit=new_fit,
            main_effect_params=new_main_effect_params,
            configurable_parameters=new_params,
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
        if config:
            self.config = default_config.merge_in_dict(config)
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

    def prepare_data(self, data: pd.DataFrame) -> Tuple[Features, Coords]:
        """
        Prepare data for modeling. Wraps the abstract method _prepare_data so
        to enforce that observed variable in features.
        """
        data_copy = (
            data.rename(columns=self.config.column_map)
            if self.config.column_map
            else data.copy()
        )
        features, coords = self._prepare_data(data_copy)

        if "obs" not in features:
            raise ValueError(
                "The model must have an observation feature named 'obs' in the configuration."
            )
        return features, coords

    @abstractmethod
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[Features, Coords]:
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


class ModelBetaBinomial(BaseModel):
    def _prepare_data(self, data: pd.DataFrame):
        # Aggregate if has not been done already
        # get nb total scores and nb correct scores
        if "n_correct" not in data.columns:
            data = (
                data.groupby(["model", "task"])
                .agg(n_correct=("score", "sum"), n_total=("score", "count"))
                .reset_index()
            )

        # map categorical levels to integer codes
        model_index = data["model"].astype("category").cat.codes
        task_index = data["task"].astype("category").cat.codes

        model_names = data["model"].astype("category").cat.categories
        task_names = data["task"].astype("category").cat.categories

        features = {
            "model_index": jnp.array(model_index),
            "task_index": jnp.array(task_index),
            "num_models": int(data["model"].nunique()),
            "num_tasks": int(data["task"].nunique()),
            "total_count": jnp.array(data["n_total"].values),
            "obs": jnp.array(
                data["n_correct"].values / data["n_total"].values
            ),  # Proportions
        }

        coords = {
            "coords": {
                "model": model_names,
                "task": task_names,
            },
            "dims": {
                "model_effects": ["model"],
                "task_effects": ["task"],
            },
        }
        return features, coords

    @classmethod
    def get_default_config(cls):
        config = super().get_default_config()
        # config = ModelConfig()

        config.configurable_parameters = [
            ParameterConfig(
                name="overall_mean",
                prior=PriorConfig(
                    distribution=dist.Normal,
                    distribution_args={"loc": 0.0, "scale": 0.3},
                ),
            ),
        ]

        config.main_effect_params = [
            "overall_mean",
            "model_effects",
        ]

        return config

    def build_model(self) -> Callable[..., Any]:
        def model(
            num_models: int,
            num_tasks: int,
            total_count: jnp.ndarray,
            model_index: jnp.ndarray,
            task_index: jnp.ndarray,
            obs=None,  # proportion of successes
        ):
            # ------------------ Global intercept -----------------------
            # overall_mean = numpyro.sample("overall_mean", dist.Normal(0, 0.3))
            overall_mean = numpyro.sample(
                **self.config.parameter_to_numpyro("overall_mean")
            )

            # ------------------ Task‑level random effects ------------

            # Separate parameter for each task
            with numpyro.plate("task_plate", num_tasks):
                task_effects = numpyro.sample("task_effects", dist.Normal(0, 0.3))

            # ------------------ Model‑level random effects -------------

            # Separate parameter for each model
            with numpyro.plate("model_plate", num_models):
                model_effects = numpyro.sample("model_effects", dist.Normal(0, 0.3))

            # For each observation compute the logit and the success probability
            # (instead of for loop do it with numpyro.plate)
            data_size = len(model_index)
            with numpyro.plate("data", data_size):
                # ---------------- Probability of success ---------------------
                # Calculate log-odds
                logits = numpyro.deterministic(
                    "logits",
                    overall_mean
                    + task_effects[task_index]
                    + model_effects[model_index],
                )

                # Convert to average probability
                avg_success_prob = numpyro.deterministic(
                    "avg_success_prob", jax.nn.sigmoid(logits)
                )

                # Overdispersion parameter (controls how much probabilities vary)
                dispersion_phi = numpyro.sample("dispersion_phi", dist.Gamma(1.0, 0.1))

                # Calculate alpha and beta for Beta distribution
                beta_alpha = avg_success_prob * dispersion_phi
                beta_beta = (1 - avg_success_prob) * dispersion_phi

                # Sample success probabilities from a Beta
                success_prob = numpyro.sample(
                    "success_prob", dist.Beta(beta_alpha, beta_beta)
                )

                # Sample observations (n_correct)
                if obs is not None:
                    # Convert proportions to counts
                    count_obs = jnp.round(obs * total_count).astype(jnp.int32)

                    # Run binomial on observed data
                    numpyro.sample(
                        "n_correct",
                        dist.Binomial(total_count, success_prob),
                        obs=count_obs,
                    )

                    # Observed is the proportion
                    numpyro.deterministic("obs", obs)

                # If no observations are provided, sample from the model (prior predictive)
                else:
                    # Generate count predictions
                    count_pred = numpyro.sample(
                        "n_correct", dist.Binomial(total_count, success_prob)
                    )
                    # Convert counts to proportions
                    numpyro.deterministic("obs", count_pred / total_count)

        return model


class TwoRandomEffectsBetaBinomial(BaseModel):
    @classmethod
    def get_default_config(cls):
        config = super().get_default_config()

        config.configurable_parameters = [
            ParameterConfig(
                name="overall_mean",
                prior=PriorConfig(
                    distribution=dist.Normal,
                    distribution_args={"loc": 0.0, "scale": 0.3},
                ),
            ),
            ParameterConfig(
                name="random_1_effects",
                prior=PriorConfig(
                    distribution=dist.Normal,
                    distribution_args={"loc": 0.0, "scale": 0.3},
                ),
            ),
            ParameterConfig(
                name="random_2_effects",
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

        config.main_effect_params = [
            "overall_mean",
            "random_1_effects",
        ]

        return config

    def _prepare_data(self, data: pd.DataFrame):
        # Aggregate if has not been done already
        # get nb total scores and nb correct scores
        if "n_correct" not in data.columns:
            data = (
                data.groupby(["random_1", "random_2"])
                .agg(n_correct=("score", "sum"), n_total=("score", "count"))
                .reset_index()
            )

        # map categorical levels to integer codes
        random_1_index = data["random_1"].astype("category").cat.codes
        random_2_index = data["random_2"].astype("category").cat.codes

        random_1_names = data["random_1"].astype("category").cat.categories
        random_2_names = data["random_2"].astype("category").cat.categories

        features = {
            "random_1_index": jnp.array(random_1_index),
            "random_2_index": jnp.array(random_2_index),
            "num_random_1": int(data["random_1"].nunique()),
            "num_random_2": int(data["random_2"].nunique()),
            "total_count": jnp.array(data["n_total"].values),
            "obs": jnp.array(
                data["n_correct"].values / data["n_total"].values
            ),  # Proportions
        }

        coords = {
            "coords": {
                "random_1": random_1_names,
                "random_2": random_2_names,
            },
            "dims": {
                "random_1_effects": ["random_1"],
                "random_2_effects": ["random_2"],
            },
        }
        return features, coords

    def build_model(self) -> Callable[..., Any]:
        def model(
            num_random_1: int,
            num_random_2: int,
            total_count: jnp.ndarray,
            random_1_index: jnp.ndarray,
            random_2_index: jnp.ndarray,
            obs=None,  # proportion of successes
        ):
            # ------------------ Global intercept -----------------------
            # overall_mean = numpyro.sample("overall_mean", dist.Normal(0, 0.3))
            overall_mean = numpyro.sample(
                **self.config.parameter_to_numpyro("overall_mean")
            )

            # ------------------ Task‑level random effects ------------

            # Separate parameter for each task
            with numpyro.plate("random_2_plate", num_random_2):
                random_2_effects = numpyro.sample(
                    **self.config.parameter_to_numpyro("random_2_effects")
                )

            # ------------------ Model‑level random effects -------------

            # Separate parameter for each model
            with numpyro.plate("random_1_plate", num_random_1):
                random_1_effects = numpyro.sample(
                    **self.config.parameter_to_numpyro("random_1_effects")
                )

            # For each observation compute the logit and the success probability
            # (instead of for loop do it with numpyro.plate)
            data_size = len(random_1_index)
            with numpyro.plate("data", data_size):
                # ---------------- Probability of success ---------------------
                # Calculate log-odds
                logits = numpyro.deterministic(
                    "logits",
                    overall_mean
                    + random_2_effects[random_2_index]
                    + random_1_effects[random_1_index],
                )

                # Convert to average probability
                avg_success_prob = numpyro.deterministic(
                    "avg_success_prob", jax.nn.sigmoid(logits)
                )

                # Overdispersion parameter (controls how much probabilities vary)
                dispersion_phi = numpyro.sample(
                    **self.config.parameter_to_numpyro("dispersion_phi")
                )

                # Calculate alpha and beta for Beta distribution
                beta_alpha = avg_success_prob * dispersion_phi
                beta_beta = (1 - avg_success_prob) * dispersion_phi

                # Sample success probabilities from a Beta
                success_prob = numpyro.sample(
                    "success_prob", dist.Beta(beta_alpha, beta_beta)
                )

                # Sample observations (n_correct)
                if obs is not None:
                    # Convert proportions to counts
                    count_obs = jnp.round(obs * total_count).astype(jnp.int32)

                    # Run binomial on observed data
                    numpyro.sample(
                        "n_correct",
                        dist.Binomial(total_count, success_prob),
                        obs=count_obs,
                    )

                    # Observed is the proportion
                    numpyro.deterministic("obs", obs)

                # If no observations are provided, sample from the model (prior predictive)
                else:
                    # Generate count predictions
                    count_pred = numpyro.sample(
                        "n_correct", dist.Binomial(total_count, success_prob)
                    )
                    # Convert counts to proportions
                    numpyro.deterministic("obs", count_pred / total_count)

        return model


class ModelBetaBinomialwSetup(BaseModel):
    def _prepare_data(self, data: pd.DataFrame):
        # Aggregate if has not been done already
        # get nb total scores and nb correct scores
        if "n_correct" not in data.columns:
            data = (
                data.groupby(["model", "task", "setup"])
                .agg(n_correct=("score", "sum"), n_total=("score", "count"))
                .reset_index()
            )

        # map categorical levels to integer codes
        model_index = data["model"].astype("category").cat.codes
        task_index = data["task"].astype("category").cat.codes
        setup_index = data["setup"].astype("category").cat.codes

        model_names = data["model"].astype("category").cat.categories
        task_names = data["task"].astype("category").cat.categories
        setup_names = data["setup"].astype("category").cat.categories

        features = {
            "model_index": jnp.array(model_index),
            "task_index": jnp.array(task_index),
            "setup_index": jnp.array(setup_index),
            "num_models": int(data["model"].nunique()),
            "num_tasks": int(data["task"].nunique()),
            "num_setups": int(data["setup"].nunique()),
            "total_count": jnp.array(data["n_total"].values),
            "obs": jnp.array(
                data["n_correct"].values / data["n_total"].values
            ),  # Proportions
        }

        coords = {
            "coords": {
                "model": model_names,
                "task": task_names,
                "setup": setup_names,
            },
            "dims": {
                "model_effects": ["model"],
                "task_effects": ["task"],
                "setup_effects": ["setup"],
            },
        }
        return features, coords

    @classmethod
    def get_default_config(cls):
        config = super().get_default_config()
        # config = ModelConfig()

        config.configurable_parameters = [
            ParameterConfig(
                name="overall_mean",
                prior=PriorConfig(
                    distribution=dist.Normal,
                    distribution_args={"loc": 0.0, "scale": 0.3},
                ),
            ),
        ]

        config.main_effect_params = [
            "overall_mean",
            "model_effects",
            "setup_effects",
        ]

        config.column_map = {}

        return config

    def build_model(self) -> Callable[..., Any]:
        def model(
            num_models: int,
            num_tasks: int,
            num_setups: int,
            total_count: jnp.ndarray,
            model_index: jnp.ndarray,
            task_index: jnp.ndarray,
            setup_index: jnp.ndarray,
            obs=None,  # proportion of successes
        ):
            # ------------------ Global intercept -----------------------
            # overall_mean = numpyro.sample("overall_mean", dist.Normal(0, 0.3))
            overall_mean = numpyro.sample(
                **self.config.parameter_to_numpyro("overall_mean")
            )

            # ------------------ Task‑level random effects ------------

            # Separate parameter for each task
            with numpyro.plate("task_plate", num_tasks):
                task_effects = numpyro.sample("task_effects", dist.Normal(0, 0.3))

            # ------------------ Model‑level random effects -------------

            # Separate parameter for each model
            with numpyro.plate("model_plate", num_models):
                model_effects = numpyro.sample("model_effects", dist.Normal(0, 0.3))

            # ------------------ Setup‑level random effects -------------

            # Separate parameter for each model
            with numpyro.plate("setup_plate", num_setups):
                setup_effects = numpyro.sample("setup_effects", dist.Normal(0, 0.3))

            data_size = len(model_index)
            with numpyro.plate("data", data_size):
                # ---------------- Probability of success ---------------------

                # Calculate log-odds
                logits = numpyro.deterministic(
                    "logits",
                    overall_mean
                    + task_effects[task_index]
                    + model_effects[model_index]
                    + setup_effects[setup_index],
                )

                # Convert to average probability
                avg_success_prob = numpyro.deterministic(
                    "avg_success_prob", jax.nn.sigmoid(logits)
                )

                # Overdispersion parameter (controls how much probabilities vary)
                dispersion_phi = numpyro.sample("dispersion_phi", dist.Gamma(1.0, 0.1))

                # Calculate alpha and beta for Beta distribution
                beta_alpha = avg_success_prob * dispersion_phi
                beta_beta = (1 - avg_success_prob) * dispersion_phi

                # Sample success probabilities from a Beta
                success_prob = numpyro.sample(
                    "success_prob", dist.Beta(beta_alpha, beta_beta)
                )

                # Sample observations (n_correct)
                if obs is not None:
                    # Convert proportions to counts
                    count_obs = jnp.round(obs * total_count).astype(jnp.int32)

                    # Run binomial on observed data
                    numpyro.sample(
                        "n_correct",
                        dist.Binomial(total_count, success_prob),
                        obs=count_obs,
                    )

                    # Observed is the proportion
                    numpyro.deterministic("obs", obs)

                # If no observations are provided, sample from the model (prior predictive)
                else:
                    # Generate count predictions
                    count_pred = numpyro.sample(
                        "n_correct", dist.Binomial(total_count, success_prob)
                    )
                    # Convert counts to proportions
                    numpyro.deterministic("obs", count_pred / total_count)

        return model


class ModelBinomial(BaseModel):
    def _prepare_data(self, data: pd.DataFrame):
        # Aggregate if has not been done already
        if "n_correct" not in data.columns:
            data = (
                data.groupby(["model", "task"])
                .agg(n_correct=("score", "sum"), n_total=("score", "count"))
                .reset_index()
            )  #

        # map categorical levels to integer codes
        model_index = data["model"].astype("category").cat.codes
        task_index = data["task"].astype("category").cat.codes

        model_names = data["model"].astype("category").cat.categories
        task_names = data["task"].astype("category").cat.categories

        features = {
            "model_index": jnp.array(model_index),
            "task_index": jnp.array(task_index),
            "num_models": int(data["model"].nunique()),
            "num_tasks": int(data["task"].nunique()),
            "total_count": jnp.array(data["n_total"].values),
            "obs": jnp.array(
                data["n_correct"].values / data["n_total"].values
            ),  # Proportions
        }

        coords = {
            "coords": {
                "model": model_names,
                "task": task_names,
            },
            "dims": {
                "model_effects": ["model"],
                "task_effects": ["task"],
            },
        }
        return features, coords

    @classmethod
    def get_default_config(cls):
        config = super().get_default_config()
        # config = ModelConfig()

        config.configurable_parameters = [
            ParameterConfig(
                name="overall_mean",
                prior=PriorConfig(
                    distribution=dist.Normal,
                    distribution_args={"loc": 0.0, "scale": 0.3},
                ),
            ),
        ]

        config.main_effect_params = [
            "overall_mean",
            "model_effects",
        ]

        return config

    def build_model(self) -> Callable[..., Any]:
        def model(
            num_models: int,
            num_tasks: int,
            total_count: jnp.ndarray,
            model_index: jnp.ndarray,
            task_index: jnp.ndarray,
            obs=None,  # proportion of successes
        ):
            # ------------------ Global intercept -----------------------
            # overall_mean = numpyro.sample("overall_mean", dist.Normal(0, 0.3))
            overall_mean = numpyro.sample(
                **self.config.parameter_to_numpyro("overall_mean")
            )

            # ------------------ Task‑level random effects ------------

            # Separate parameter for each task
            with numpyro.plate("task_plate", num_tasks):
                task_effects = numpyro.sample("task_effects", dist.Normal(0, 0.3))

            # ------------------ Model‑level random effects -------------

            # Separate parameter for each model
            with numpyro.plate("model_plate", num_models):
                model_effects = numpyro.sample("model_effects", dist.Normal(0, 0.3))

            # For each observation compute the logit and the success probability
            # (instead of for loop do it with numpyro.plate)
            data_size = len(model_index)
            with numpyro.plate("data", data_size):
                # ---------------- Probability of success ---------------------

                # Calculate log-odds
                logits = numpyro.deterministic(
                    "logits",
                    overall_mean
                    + task_effects[task_index]
                    + model_effects[model_index],
                )

                # Sample success probabilities from a Beta
                success_prob = numpyro.deterministic(
                    "success_prob", jax.nn.sigmoid(logits)
                )

                # Sample observations (n_correct)
                if obs is not None:
                    # Convert proportions to counts
                    count_obs = jnp.round(obs * total_count).astype(jnp.int32)

                    # Run binomial on observed data
                    numpyro.sample(
                        "n_correct",
                        dist.Binomial(total_count, success_prob),
                        obs=count_obs,
                    )

                    # Observed is the proportion
                    numpyro.deterministic("obs", obs)

                # If no observations are provided, sample from the model (prior predictive)
                else:
                    # Generate count predictions
                    count_pred = numpyro.sample(
                        "n_correct", dist.Binomial(total_count, success_prob)
                    )
                    # Convert counts to proportions
                    numpyro.deterministic("obs", count_pred / total_count)

        return model
