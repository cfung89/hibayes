import argparse
import os

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description="Load CSV files")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV file to load",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="dir to write the CSV file to",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    df = pd.read_csv(args.csv)
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    df.to_csv(args.out + "data.csv", index=False)
