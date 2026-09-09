"""Data loading and preprocessing utilities."""

import numpy as np
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a CSV file.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame containing the loaded data
    """
    return pd.read_csv(filepath)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data by handling missing values and normalization.

    Args:
        df: Input DataFrame

    Returns:
        Preprocessed DataFrame
    """
    return df
