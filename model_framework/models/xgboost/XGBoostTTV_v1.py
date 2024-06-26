import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from utils import TimeSeries


class XGBoostTTV_v1:
    # Regression hyperparameters:
    __kwargs_hyperparams = dict(
        base_score=0.50,                # base "average" to build regression trees from
        booster='gbtree',               # the gradient booster used
        n_estimators=1200,              # number of estimators (trees)
        early_stopping_rounds=800,      # early stopping rounds if loss plateaus
        objective="reg:squarederror",   # loss function to use
        max_depth=4,                    # maximum (tree) depth
        learning_rate=0.005             # learning rate for regressor
    )

    def __init__(self, time_series: TimeSeries, read_from_stub=None, write_to_stub=None):
        """
        Creates an XGBoostTTV object (uses XGBoostRegressor) to model the given TimeSeries (time_series)
        data.

        :param time_series:     the TimeSeries to model.
        """
        # Storing this TimeSeries:
        self.time_series = time_series
        # Creating a list of features (as column names) for the dataset:
        self.features = time_series.features + time_series.lags
        # Target to predict (as a column name):
        self.target = time_series.value_name

        self.model_name = "XGBoostTTV_v1"
        self.read_from_stub = read_from_stub
        self.write_to_stub = write_to_stub

        # Initializing a saved model with scores and cross-validation predictions:
        self.regressor = None
        if (read_from_stub is not None) and os.path.exists(read_from_stub):
            self.regressor = xgb.XGBRegressor(**self.__kwargs_hyperparams)
            self.regressor.load_model(read_from_stub)

    def train(self, epochs=None):
        """
        Trains the XGBoostRegressor model on all (training) data according to hyperparameters defined.

        :return:        None.
        """
        # Creating a regressor that takes in ALL the available data:
        if self.regressor is None:
            regressor = xgb.XGBRegressor(**self.__kwargs_hyperparams)
        else:
            regressor = self.regressor

        # Initializing predictions and scores:
        predictions = []
        scores = []

        # Creating training and testing sets:
        X_train = self.time_series.df_split_train[self.features]
        y_train = self.time_series.df_split_train[self.target]

        X_test = self.time_series.df_split_test[self.features]
        y_test = self.time_series.df_split_test[self.target]

        # Setting verbose value:
        if self.time_series.verbose:
            verbose = 100
        else:
            verbose = False

        # Fitting regressor to dataset:
        regressor.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_test, y_test)],
                      verbose=verbose)

        # Storing the training information:
        #   y_pred:     prediction of regressor in the testing set.
        #   score:      overall error (lower is better)
        y_pred = regressor.predict(X_test)
        predictions.append(y_pred)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(score)

        # Saving this regressor as an instance attribute:
        self.regressor = regressor
        # Saving this regressor into a file:
        if (self.write_to_stub is not None) and os.path.exists(self.write_to_stub):
            self.regressor.save_model(self.write_to_stub)
            print(f"Saving trained model to {self.write_to_stub}.")
        return

    def predict(self, custom_df=None, datetime_name=None, value_name=None):
        """
        Fits the model onto the future data in the TimeSeries using the trained model (by default)
        Requires that the model has been trained via the train() method first.

        :param: custom_df:      predict on a custom dataset.
        :param: datetime_name:  name of the DateTime column, if custom_df is provided.
        :param: value_name:     name of the Target column, if custom_df is provided.
        :return:                predictions of the model (as an array).
        """
        if (self.regressor is None) and (self.read_from_stub is None):
            raise Exception(f"Regressor has not been trained yet.")

        # If a custom dataframe is not provided, set future_dataset to TimeSeries.df_future_only:
        if custom_df is None:
            if self.time_series.df_future_only.shape[0] > self.time_series.lag_min:
                print(f"SoftWarn: Future window size ({self.time_series.df_future_only.shape[0]}) is larger than "
                      f"the smallest lag ({self.time_series.lag_min}).")
            # Filling all NaN values with 0 (if they exist), initializing dataset:
            future_dataset = self.time_series.df_future_only.fillna(0).copy()
            # Dropping DateTime column (the predictor does not accept the pd.datetime64 format) and
            #   the Target column (this is what the model is trying to predict):
            future_dataset = future_dataset.drop(columns=[self.time_series.datetime_name,
                                                          self.time_series.value_name])

        # Otherwise, use the custom dataframe:
        else:
            if custom_df.shape[0] > self.time_series.lag_min:
                print(f"SoftWarn: Future window size ({custom_df.shape[0]}) is larger than "
                      f"the smallest lag ({self.time_series.lag_min}).")
            if (datetime_name is None) or (value_name is None):
                raise IndexError("Please provide the DateTime name and Target name.")
            # Filling all NaN values with 0 (if they exist), initializing dataset:
            future_dataset = custom_df.fillna(0).copy()
            # Dropping DateTime column (the predictor does not accept the pd.datetime64 format) and
            #   the Target column (this is what the model is trying to predict):
            future_dataset = future_dataset.drop(columns=[datetime_name,
                                                          value_name])
        # Making prediction:
        prediction = self.regressor.predict(future_dataset)

        return prediction
