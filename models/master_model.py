import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from utils import TimeSeries


class MasterModel:
    # Here's how the TimeSeries data is used:
    #   time_series.df_augmented:       typically unused, except for cross-validation models.
    #   time_series.features:           used as features when predicting.
    #   time_series.lags:               used as features when predicting.
    #   time_series.lag_min:            used as a benchmark for the maximum recommended prediction window.
    #   time_series.value_name:         the Target to predict.
    #   time_series.df_split_<>:        splits used for training, testing and validation (depending on model).
    #   time_series.future_<>:          dataset onto which the model predicts (depending on model).
    def __init__(self, time_series: TimeSeries, model: str):
        """
        MasterModel uses a TimeSeries, and a provided model, to train and forecast.

        :param time_series:     TimeSeries object [must be fully prepared, use prepare_from_scratch()].
        :param model:           the model used, as a str; currently supports: "...".
        """
        # Storing the time_series and model_name:
        self.time_series = time_series
        self.model_name = model

        # Creating a list of features (as column names) for the dataset:
        self.features = time_series.features + time_series.lags
        # Target to predict (as a column name):
        self.target = time_series.value_name

