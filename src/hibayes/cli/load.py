import argparse
import pathlib

from ..analysis import AnalysisConfig, load_data
from ..ui import ModellingDisplay


def run_load(args):
    config = AnalysisConfig.from_yaml(args.config)
    display = ModellingDisplay()
    df = load_data(config=config.data_loader, display=display)
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        args.out + "/data.parquet",
    )


def main():
    parser = argparse.ArgumentParser(description="Load data hibayes.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration file (YAML format). See examples/*/*.yaml for examples",
    )
    parser.add_argument(
        "--out", required=True, help="Where to write the DVC tracking files"
    )

    args = parser.parse_args()
    run_load(args)


if __name__ == "__main__":
    main()
