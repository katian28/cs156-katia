"""Plotting utilities for visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_distribution(data: pd.Series, title: str = "Distribution") -> None:
    """Plot the distribution of a series.

    Args:
        data: Series to plot
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def plot_scatter(x: pd.Series, y: pd.Series, title: str = "Scatter Plot") -> None:
    """Plot a scatter plot.

    Args:
        x: X-axis data
        y: Y-axis data
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
