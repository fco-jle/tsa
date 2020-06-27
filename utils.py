import os
import sys

import matplotlib
import mplcyberpunk
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.indexes.datetimes import DatetimeIndex


class DataSamples:
    @staticmethod
    def load(filename, index_col=None, parse_dates=True, drop_na=True, index_freq=None):
        df = pd.read_csv(Config.path_from_data(f'{filename}.csv'), parse_dates=parse_dates)

        if index_col is not None:
            df = df.set_index(index_col)

        if drop_na:
            df = df.dropna()

        if index_freq is not None:
            assert isinstance(df.index, DatetimeIndex)
            df.index.freq = index_freq

        if isinstance(df.index, DatetimeIndex):
            df = df.sort_index()

        return df


class PlotStyle:
    @staticmethod
    def cyber():
        assert mplcyberpunk.__name__ in sys.modules
        plt.style.use("cyberpunk")
        matplotlib.rcParams['lines.linewidth'] = 1
        matplotlib.rcParams['lines.markersize'] = 2
        std_color = matplotlib.rcParams['axes.prop_cycle']
        std_color = [list(c.values())[0] for c in std_color]
        return std_color


class Config:
    repo_path = os.path.dirname(os.path.abspath(__file__))
    examples_data_path = os.path.join(repo_path, 'data_samples')

    @classmethod
    def path_from_root(cls, path):
        return os.path.join(cls.repo_path, path)

    @classmethod
    def path_from_data(cls, path):
        return os.path.join(cls.examples_data_path, path)
