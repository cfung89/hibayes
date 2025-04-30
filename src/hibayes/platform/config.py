import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class PlatformConfig:
    device_type: str = "cpu"  # Device type (cpu, gpu, tpu)
    num_devices: int | None = None  # Number of devices to use (None = auto-detect)

    def __post_init__(self):
        # Auto-detect number of devices if not explicitly provided
        if self.num_devices is None:
            if self.device_type == "cpu":
                self.num_devices = os.cpu_count()
            else:
                raise NotImplementedError(
                    f"{self.device_type.upper()} support is not yet implemented. Please use CPU for now."
                )

    def merge_in_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Merge a dictionary into the existing configuration.
        """
        if not config_dict:
            return

        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Invalid configuration for {self.__class__.__name__} key: {key}"
                )

    @classmethod
    def from_yaml(cls, path: str) -> "PlatformConfig":
        """
        Load configuration from a yaml file.
        """
        with open(path, "r") as f:
            config: dict = yaml.safe_load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict | None) -> "PlatformConfig":
        """
        Load configuration from a dictionary.
        """
        if config is None:
            return cls()
        return cls(
            device_type=config.get("device_type", "cpu"),
            num_devices=config.get("num_devices", None),
        )
