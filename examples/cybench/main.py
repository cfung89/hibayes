"""
If you do not want to use DVC this file can be used to run the analysis
"""

import argparse
import os

from hibayes.cli.communicate import run_communicate
from hibayes.cli.model import run_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit statistical models and run quality checks using hibayes on cybench logs."
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

    return parser.parse_args()


def main():
    args = parse_args()

    run_model(args)  # will save the results in the out dir.

    # add the analysis state loc to args
    args.analysis_state = os.path.join(args.out)
    run_communicate(args)  # will save the results in the out dir.


if __name__ == "__main__":
    main()
