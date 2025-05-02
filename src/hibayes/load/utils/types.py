from typing import Any, Dict, List, Union

import pandas as pd


def check_mixed_types(df: pd.DataFrame) -> None:
    """
    Check if the DataFrame contains mixed types in any column.

    Args:
        df (pd.DataFrame): The DataFrame to check.
    """
    mixed = {
        col: df[col].dropna().map(type).unique()
        for col in df.columns
        if df[col].dropna().map(type).nunique() > 1
    }
    if mixed:
        raise TypeError(
            "Mixed Python types detected in the following cols — fix them before writing Parquet:\n"
            + "\n".join(f"- {c}: {sorted(v)}" for c, v in mixed.items())
            + "If you have added custom extractors, please check that they types are consistent.\n"
        )
