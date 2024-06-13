import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from utils import TimeSeries
from models import MasterModel


class XGBoostTTV_v1(MasterModel):
    # Hyperparameters:
    #   Number of estimator (trees) to use:
    __N_ESTIMATORS = 1000
    #   Early stopping rounds:
    __EARLY_STOPPING = 300
    #   Maximum (tree) depth:
    __MAX_DEPTH = 3
    #   Learning rate for regressor:
    __LEARNING_RATE = 0.01
    #   Loss function for regressor (as per XGBoostRegressor):
    __LOSS_FUNCTION = "reg:squarederror"

    def __init__(self, time_series: TimeSeries):
        """
        Creates an XGBoostTTV object (uses XGBoostRegressor) to model the given TimeSeries (time_series)
        data.

        :param time_series:     the TimeSeries to model.
        """
        super().__init__(time_series, "XGBoostTTV_v1")

        # Initializing a saved model with scores and cross-validation predictions:
        self.regressor = None
        self.scores = None
        self.predictions = None

    def train(self):
        """
        Trains the XGBoostRegressor model on all (training) data according to hyperparameters defined.

        :return:        None.
        """
        # Creating a regressor that takes in ALL the available data:
        regressor = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                                     n_estimators=self.__N_ESTIMATORS,
                                     early_stopping_rounds=self.__EARLY_STOPPING,
                                     objective=self.__LOSS_FUNCTION,
                                     max_depth=self.__MAX_DEPTH,
                                     learning_rate=self.__LEARNING_RATE)

        # Initializing predictions and scores:
        predictions = []
        scores = []

        # Creating training and testing sets:
        X_train = self.time_series.df_split_train[self.features]
        y_train = self.time_series.df_split_train[self.target]

        X_test = self.time_series.df_split_test[self.features]
        y_test = self.time_series.df_split_test[self.target]

        regressor.fit(X_train, y_train,
                      eval_set=[(X_train, y_train)],
                      verbose=100)

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

    def predict(self):
        """
        Fits the model onto the future data in the TimeSeries using the trained model.
        Requires that the model has been trained via the train() method first.

        :return:        predictions of the model.
        """
        if self.regressor is None:
            raise Exception(f"Regressor has not been trained yet.")
        # If the future prediction window is larger than the smallest lag:
        if self.time_series.df_future_only.shape[0] > self.time_series.lag_min:
            print(f"SoftWarn: Future window size ({self.time_series.df_future_only.shape[0]}) is larger than"
                  f"the smallest lag ({self.time_series.lag_min}).")

        # Filling all NaN values with 0 (if they exist):
        future_dataset = self.time_series.df_future_only.fillna(0)
        # Dropping DateTime column (the predictor does not accept the pd.datetime64 format) and
        #   the Target column (this is what the model is trying to predict):
        future_dataset = future_dataset.drop(columns=[self.time_series.datetime_name,
                                                      self.time_series.value_name])
        # Making prediction:
        prediction = self.regressor.predict(future_dataset)

        return prediction
