import argparse
import pathlib

import pandas as pd

from ..analysis import AnalysisConfig, model
from ..ui import ModellingDisplay


def _read_df(path: pathlib.Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".jsonl", ".json"):
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file extension: {ext}")


def run_model(args):
    config = AnalysisConfig.from_yaml(args.config)
    display = ModellingDisplay()
    out = pathlib.Path(args.out)

    out.mkdir(parents=True, exist_ok=True)
    df = _read_df(pathlib.Path(args.data))

    analysis_state = model(
        data=df,
        model_config=config.models,
        checker_config=config.checkers,
        platform_config=config.platform,
        display=display,
    )
    analysis_state.save(path=out)


def main():
    parser = argparse.ArgumentParser(
        description="Fit statistical models and run quality checks using hibayes."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration file (YAML format). See examples/*/*.yaml for examples",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the processed data file (Parquet, csv, jsonl format accepted).",
    )
    parser.add_argument(
        "--out", required=True, help="dir path to write the DVC tracking files"
    )

    args = parser.parse_args()

    run_model(args)


if __name__ == "__main__":
    main()
