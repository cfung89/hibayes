import datetime
import logging
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, Iterator, List, Optional

import yaml
from hibayes.utils import init_logger

from ..extractors import (
    BaseMetadataExtractor,
    MetadataExtractor,
    TokenExtractor,
    ToolsExtractor,
)

logger = init_logger()


@dataclass
class DataLoaderConfig:
    """Configuration for the log processor."""

    # Define a list of default extractors that are used if nothing specified
    DEFAULT_EXTRACTORS = ["base"]

    # Define all available extractors with their keys
    AVAILABLE_EXTRACTORS = {
        "base": BaseMetadataExtractor(),
        "tools": ToolsExtractor(),
        "tokens": TokenExtractor(),
    }

    # Configuration properties
    enabled_extractors: List[str]
    custom_extractors: List[MetadataExtractor]

    files_to_process: List[str]
    cache_path: str | None = None
    output_dir: str | None = None
    max_workers: int = 10
    batch_size: int = 1000
    cutoff: datetime.datetime | None = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DataLoaderConfig":
        """
        Create a DataLoaderConfig from a YAML file

        The YAML should have this structure:
        ```yaml
        extractors:
          enabled:
            - base
            - domain
            - tools
          custom:
            path: "my_custom_extractors.py"
            classes:
              - CustomMetadataExtractor
              - PromptAnalysisExtractor
        ```
        """
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any] | None) -> "DataLoaderConfig":
        if config_dict is None:
            config_dict = {}
        # Get enabled extractors or use default if not specified
        enabled = config_dict.get("extractors", {}).get("enabled")
        if not enabled:
            enabled = cls.DEFAULT_EXTRACTORS

        # get paths for logs and processed data
        config_paths = config_dict.get("paths", {})
        files_to_process = config_paths.get("files_to_process", [])
        cache_path = config_paths.get("cache_path")
        output_dir = config_paths.get("output_dir")

        custom_extractors = []
        custom_config = config_dict.get("extractors", {}).get("custom", {})
        if custom_config:
            import importlib.util

            module_path = custom_config.get("path")
            if module_path:
                spec = importlib.util.spec_from_file_location(
                    "custom_extractors", module_path
                )
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    for class_name in custom_config.get("classes", []):
                        if hasattr(module, class_name):
                            custom_extractors.append(getattr(module, class_name)())

        return cls(
            enabled_extractors=enabled,
            custom_extractors=custom_extractors,
            files_to_process=files_to_process,
            cache_path=cache_path,
            output_dir=output_dir,
        )

    @classmethod
    def create_default(cls) -> "DataLoaderConfig":
        """Create a default configuration."""
        return cls()
