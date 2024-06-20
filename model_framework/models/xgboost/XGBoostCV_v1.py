import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from utils import TimeSeries


class XGBoostCV_v1:
    # Regression hyperparameters:
    __kwargs_hyperparams = dict(
        base_score=0.50,                # base "average" to build regression trees from
        booster='gbtree',               # the gradient booster used
        n_estimators=400,               # number of estimators (trees)
        early_stopping_rounds=300,      # early stopping rounds if loss plateaus
        objective="reg:squarederror",   # loss function to use
        max_depth=4,                    # maximum (tree) depth
        learning_rate=0.01              # learning rate for regressor
    )
    # Number of splits to use for cross-validation:
    __SPLITS = 8

    def __init__(self, time_series: TimeSeries):
        """
        Creates an XGBoostCV object (which uses cross-validation with XGBoostRegressor) to model the
        given TimeSeries (time_series) data.

        :param time_series:     the TimeSeries to model.
        """
        # Storing this TimeSeries:
        self.time_series = time_series
        # Creating a list of features (as column names) for the dataset:
        self.features = time_series.features + time_series.lags
        # Target to predict (as a column name):
        self.target = time_series.value_name

        self.model_name = "XGBoostCV_v1"
        # Creating a TimeSeriesSplit (for cross-validation) with default test_size (gap 0):
        self.time_series_split = TimeSeriesSplit(n_splits=self.__SPLITS)

        # Creating a dataset for this model (this will be the training set from the TimeSeries):
        self.dataset = self.time_series.df_split_train

        # Initializing a saved model with scores and cross-validation predictions:
        self.regressor = None
        self.scores = None
        self.predictions = None

    def train(self):
        """
        Trains the XGBoostRegressor model with cross-validation according to hyperparameters defined.

        :return:        None.
        """
        # Creating the Regressor model:
        regressor = xgb.XGBRegressor(**self.__kwargs_hyperparams)

        # Initializing predictions and scores:
        predictions = []
        scores = []

        # Iterating through the (self.__SPLITS)-fold cross-validation training process:
        for train_idx, val_idx in self.time_series_split.split(self.dataset):
            # Creating this train and test instance:
            train = self.dataset.iloc[train_idx]
            test = self.dataset.iloc[val_idx]

            # Extracting the training and testing features:
            #   Extracting training features:
            X_train = train[self.features]
            y_train = train[self.target]
            #   Extracting testing features:
            X_test = test[self.features]
            y_test = test[self.target]

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
        # Saving the scores and predictions:
        self.scores = scores
        self.predictions = predictions
        return

    def predict(self, custom_df=None):
        """
        Fits the model onto the future data in the TimeSeries using the trained model.
        Requires that the model has been trained via the train() method first.

        :param: custom_df:      predict on a custom dataset. must be prepared using TimeSeries.df_create_future()
        :return:                predictions of the model.
        """
        if self.regressor is None:
            raise Exception(f"Regressor has not been trained yet.")
        # If the future prediction window is larger than the smallest lag:
        if custom_df is None:
            if self.time_series.df_future_only.shape[0] > self.time_series.lag_min:
                print(f"SoftWarn: Future window size ({self.time_series.df_future_only.shape[0]}) is larger than "
                      f"the smallest lag ({self.time_series.lag_min}).")
        else:
            if custom_df.shape[0] > self.time_series.lag_min:
                print(f"SoftWarn: Future window size ({custom_df.shape[0]}) is larger than "
                      f"the smallest lag ({self.time_series.lag_min}).")

        # Filling all NaN values with 0 (if they exist), initializing dataset:
        if custom_df is not None:
            future_dataset = custom_df.fillna(0).copy()
        else:
            future_dataset = self.time_series.df_future_only.fillna(0).copy()
        # Dropping DateTime column (the predictor does not accept the pd.datetime64 format) and
        #   the Target column (this is what the model is trying to predict):
        future_dataset = future_dataset.drop(columns=[self.time_series.datetime_name,
                                                      self.time_series.value_name])
        # Making prediction:
        prediction = self.regressor.predict(future_dataset)

        return prediction