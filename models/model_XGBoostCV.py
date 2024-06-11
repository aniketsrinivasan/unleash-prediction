import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


class ModelXGBoostCV:
    # Cross-Validation hyperparameters:
    #   Number of splits to use for cross-validation:
    __SPLITS = 10
    #   Number of estimator (trees) to use:
    __N_ESTIMATORS = 500
    #   Early stopping rounds:
    __EARLY_STOPPING = 300
    #   Maximum (tree) depth:
    __MAX_DEPTH = 3
    #   Learning rate for regressor:
    __LEARNING_RATE = 0.01

    def __init__(self, dataframe, features: list[str], target: str):
        # Saving pd.DataFrame and TimeSeriesSplit objects as instance attributes:
        self.dataframe = dataframe
        # Saving features and target:
        self.features = features
        self.target = target

        # Initializing regressor models:
        self.time_series_split = None
        self.regressor_cv = None
        self.regressor_all = None

    def train_cv(self, test_size, n_splits=__SPLITS):
        # Initializing the TimeSeriesSplit object (used for cross-validation):
        self.time_series_split = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        preds = []
        scores = []
        for train_idx, val_idx in self.time_series_split.split(self.dataframe):
            # Creating this train and test instance:
            train = self.dataframe.iloc[train_idx]
            test = self.dataframe.iloc[val_idx]

            # Extracting the training and testing features:
            #   Extracting training features:
            X_train = train[self.features]
            y_train = train[self.target]
            #   Extracting testing features:
            X_test = test[self.features]
            y_test = test[self.target]

            # Creating the XGBoost Regressor model:
            regressor = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                                         n_estimators=self.__N_ESTIMATORS,
                                         early_stopping_rounds=self.__EARLY_STOPPING,
                                         objective='reg:squarederror',
                                         max_depth=self.__MAX_DEPTH,
                                         learning_rate=self.__LEARNING_RATE)

            # Fitting regressor to dataset:
            regressor.fit(X_train, y_train,
                          eval_set=[(X_train, y_train), (X_test, y_test)],
                          verbose=100)

            # Storing the training information:
            #   y_pred:     prediction of regressor in the testing set.
            #   score:      overall error (lower is better)
            y_pred = regressor.predict(X_test)
            preds.append(y_pred)
            score = np.sqrt(mean_squared_error(y_test, y_pred))
            scores.append(score)

            # Saving this regressor as an instance attribute:
            self.regressor_cv = regressor

        return preds, scores

    def train_all(self, n_estimators, early_stopping, max_depth, learning_rate):
        X_all = self.dataframe[self.features]
        y_all = self.dataframe[self.target]

        # Creating a regressor that takes in ALL the available data:
        regressor = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                                     n_estimators=n_estimators,
                                     early_stopping_rounds=early_stopping,
                                     objective='reg:squarederror',
                                     max_depth=max_depth,
                                     learning_rate=learning_rate)
        regressor.fit(X_all, y_all,
                      eval_set=[(X_all, y_all)],
                      verbose=100)

        # Saving this regressor as an instance attribute:
        self.regressor_all = regressor

    def predict_cv(self, window_size):
        # create "future" dataset:
        #   data_create_future(dataframe, window_size, step="1h")
        # run .predict on self.regressor_cv
        # return dataset
        pass


