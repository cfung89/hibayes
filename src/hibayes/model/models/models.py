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


@dataclass
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

    def merge_in_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Merge a dictionary into the existing configuration.
        """
        if not config_dict:
            return

        for key, value in config_dict.items():
            if hasattr(self, key):
                if key == "distribution":
                    if value in self.DISTRIBUTION_MAP:
                        self.distribution = self.DISTRIBUTION_MAP[value]
                    else:
                        raise ValueError(
                            f"Unsupported distribution: {value}. Supported distributions are: {', '.join(self.DISTRIBUTION_MAP.keys())}"
                        )
                elif key == "distribution_args":
                    if isinstance(value, dict):
                        self.distribution_args.update(value)
                else:
                    raise ValueError(
                        f"Invalid configuration for {self.__class__.__name__} key: {key}"
                    )
            else:
                raise ValueError(
                    f"Invalid configuration for {self.__class__.__name__} key: {key}"
                )


@dataclass
class ParameterConfig:
    name: str  # Name of the parameter in the model
    prior: PriorConfig | None = None  # Prior distribution for the parameter

    def merge_in_dict(self, config_dict: Dict[str, Any]) -> None:
        new_param_name = config_dict.get("name")
        new_prior = config_dict.get("prior", {})

        if new_param_name != self.name:
            raise ValueError(
                f"Parameter name mismatch: '{new_param_name}' does not match '{self.name}'. You cannot update parameter names as they are called in the model."
            )
        if new_prior:
            if not self.prior:
                raise ValueError(
                    "You cannot update the prior of a parameter that does not have one."
                )
            self.prior.merge_in_dict(new_prior)


@dataclass
class ModelConfig:
    mapping_name: dict[
        str, str
    ] | None = (
        None  # the key for mapping from data columns to any required_column names
    )
    configurable_parameters: List[
        ParameterConfig
    ] | None = None  # parameters user can adjust
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

    def get_plot_params(self) -> List[str]:
        """
        Get a list of parameters to plot based on the configuration."""
        return self.main_effect_params

    def get_params(self) -> List[str]:
        """
        Get a list of all configurable parameters in the model config."""
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

    def merge_in_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Merge a dictionary into the existing configuration.
        """
        if not config_dict:
            return

        for key, value in config_dict.items():
            if hasattr(self, key):
                if key == "configurable_parameters":
                    if isinstance(value, list):
                        self.merge_parameters(value)
                    elif isinstance(value, dict):
                        self.merge_parameter(value)
                    else:
                        raise ValueError(
                            "To update parameters you need to specify either a list of dicts or a single dict where for each parameter at least the parameter name and what you would like to change is detailed."
                        )
                elif key == "fit":
                    if isinstance(value, dict):
                        self.fit.merge_in_dict(value)
                    else:
                        raise ValueError(
                            f"Invalid configuration for {self.__class__.__name__} key: {key}"
                        )
                elif key == "platform":
                    if isinstance(value, dict):
                        self.platform.merge_in_dict(value)
                    else:
                        raise ValueError(
                            f"Invalid configuration for {self.__class__.__name__} key: {key}"
                        )
                elif key == "mapping_name":
                    if isinstance(value, dict):
                        self.mapping_name = value  # do not combine, mapping dict should be user defined.
                    else:
                        raise ValueError(
                            f"Invalid configuration for {self.__class__.__name__} key: {key}"
                        )
                else:
                    raise ValueError(
                        f"Invalid configuration for {self.__class__.__name__} key: {key}"
                    )

            else:
                raise ValueError(
                    f"Invalid configuration for {self.__class__.__name__} key: {key}"
                )

    def merge_parameters(self, new_params: List[Dict[str, Any]]) -> None:
        """
        Merge new parameter configurations into the existing ones.
        """
        for param in new_params:
            if isinstance(param, dict):
                self.merge_parameter(param)
            else:
                raise ValueError(
                    f"Invalid parameter configuration: {param}. Must be a dictionary."
                )

    def merge_parameter(self, param: Dict[str, Any]) -> None:
        """
        Merge a single parameter configuration into the existing ones.
        """
        if default_param := self.get_param(param["name"]):
            default_param.merge_in_dict(param)
        else:
            raise ValueError(
                f"Parameter name '{param['name']}' not found in model configuration"
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
        self.config = self.get_default_config()
        self.config.merge_in_dict(config)

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
            data.rename(columns=self.config.mapping_name)
            if self.config.mapping_name
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

        config.mapping_name = {}

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
