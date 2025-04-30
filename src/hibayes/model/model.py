import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import pandas as pd
import yaml
from hibayes.utils import init_logger

from ..analysis_state import ModelAnalysisState
from .models import (
    BaseModel,
    ModelBetaBinomial,
    ModelBinomial,
    ModelConfig,
    ModelSampleEffects,
)

logger = init_logger()


@dataclass
class ModelsToRunConfig:
    """Configuration for the models to be run in the analysis."""

    DEFAULT_MODELS: ClassVar[List[str]] = ["ModelBetaBinomial"]
    AVAILABLE_MODELS: ClassVar[Dict[str, Type[BaseModel]]] = {
        "EmpiricalMean": None,  # Replace with actual classes when available
        "ModelMean": None,
        "ModelDomainGLM": None,
        "ModelDomainClusterGLM": None,
        "CompareVarEffects": None,
        "ModelSampleEffects": ModelSampleEffects,
        "ModelBetaBinomial": ModelBetaBinomial,
        "ModelBinomial": ModelBinomial,
    }

    # List of (model_class, model_config) tuples
    enabled_models: List[
        tuple[Type[BaseModel], Optional[Union[Dict[str, Any], ModelConfig]]]
    ] = field(
        default_factory=list,
    )

    def __post_init__(self) -> None:
        """Set up default models if none specified."""
        if not self.enabled_models and not hasattr(self, "_initialised"):
            self.enabled_models = [
                (self.AVAILABLE_MODELS[model], None)
                for model in self.DEFAULT_MODELS
                if self.AVAILABLE_MODELS[model] is not None
            ]
            self._initialised = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelsToRunConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            config: dict = yaml.safe_load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict) -> "ModelsToRunConfig":
        """Load configuration from a dictionary."""

        if config is None:
            config = {}

        enabled_models = []
        if custom_models_config := config.get("custom_models", None):
            enabled_models.extend(cls._load_custom_models(custom_models_config))

        models_config = config.get("models", None)

        # Handle different formats of model configuration
        if isinstance(models_config, list):
            # ["Model1", "Model2"] format
            for model_name in models_config:
                if (
                    model_name in cls.AVAILABLE_MODELS
                    and cls.AVAILABLE_MODELS[model_name] is not None
                ):
                    enabled_models.append(
                        (cls.AVAILABLE_MODELS[model_name], None)
                    )  # notice no config so just use default model args.
                else:
                    logger.warning(
                        f"Model {model_name} not available or not properly defined."
                    )

        elif isinstance(models_config, dict):
            # {"Model1": {config}, "Model2": {config}} format
            for model_name, model_config in models_config.items():
                if (
                    model_name in cls.AVAILABLE_MODELS
                    and cls.AVAILABLE_MODELS[model_name] is not None
                ):
                    # Here we are passing the custom args to the model config.
                    enabled_models.append(
                        (cls.AVAILABLE_MODELS[model_name], model_config)
                    )
                else:
                    logger.warning(
                        f"Model {model_name} not available or not properly defined."
                    )

        # If no models were successfully loaded, use defaults
        if not enabled_models:
            logger.info("No valid models specified, using defaults.")
            enabled_models = [
                (cls.AVAILABLE_MODELS[model], None)
                for model in cls.DEFAULT_MODELS
                if model in cls.AVAILABLE_MODELS
                and cls.AVAILABLE_MODELS[model] is not None
            ]

        return cls(
            enabled_models=enabled_models,
        )

    @staticmethod
    def _load_custom_models(
        config: dict[str, dict],
    ) -> List[tuple[Type[BaseModel], Optional[ModelConfig]]]:
        """Load custom model classes from the specified path."""
        custom_models: List[tuple[Type[BaseModel], Optional[ModelConfig]]] = []

        try:
            import importlib.util
            import inspect

            module_path = config.get("path", None)
            if module_path:
                spec = importlib.util.spec_from_file_location(
                    "custom_models", module_path
                )
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    for model_name in config.get(
                        "classes",
                    ):
                        if hasattr(module, model_name):
                            model_class = getattr(module, model_name)
                            if issubclass(model_class, BaseModel):
                                custom_models.append(
                                    (model_class, None)
                                )  # for now custom models use their default config. TODO: update to allow custom config.
                                logger.info(f"Loaded custom model: {model_name}")
                            else:
                                logger.warning(
                                    f"{model_name} is not a subclass of BaseModel."
                                )
                        else:
                            logger.warning(f"{model_name} not found in custom module.")
        except ImportError as e:
            logger.error(f"Error importing custom models: {e}")

        return custom_models

    def build_models(self, data: pd.DataFrame) -> Iterator[ModelAnalysisState]:
        """
        Building all enabled models with the provided data and their configurations.

        Parameters
        ----------
        data : pd.DataFrame
            The data to be used for model fitting

        Returns
        -------
        AnalysisState
            A generator yielding AnalysisState objects for each model.

        """
        for model_builder_cls, model_config in self.enabled_models:
            logger.info(f"Instantiating model: {model_builder_cls.name}")
            model_name = model_builder_cls.name()
            model_builder = model_builder_cls(
                config=model_config,
            )
            logger.info(f"Instantiated model: {model_name}")

            logger.debug(f"Extracting features for model: {model_name}")
            features, coords = model_builder.prepare_data(data)
            logger.info(f"Extracted features for model: {model_name}")

            yield ModelAnalysisState(
                model_name=model_name,
                model_builder=model_builder,
                features=features,
                coords=coords,
            )
