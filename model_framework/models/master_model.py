import pandas as pd
import numpy as np
from .xgboost import *
from .model_utils import validation_loss
from utils import TimeSeries


class MasterModel:
    # Here's how the TimeSeries data is used:
    #   time_series.df_augmented:       typically unused, except for cross-validation model_framework.
    #   time_series.features:           used as features when predicting.
    #   time_series.lags:               used as features when predicting.
    #   time_series.lag_min:            used as a benchmark for the maximum recommended prediction window.
    #   time_series.value_name:         the Target to predict.
    #   time_series.df_split_<>:        splits used for training, testing and validation (depending on model).
    #   time_series.future_<>:          dataset onto which the model predicts (depending on model).
    def __init__(self, time_series: TimeSeries, model_name: str,
                 read_from_stub=False, write_to_stub=False, stub_path=None):
        """
        MasterModel uses a TimeSeries, and a provided model name, to create a Model of that type.
        This acts as a wrapper function to all the Model classes defined.

        :param time_series:     TimeSeries object [must be fully prepared, use prepare_from_scratch()].
        :param model_name:      the model used, as a str; currently supports: "...".
        :param read_from_stub:  whether to load the model from a stub (stub_path).
        :param write_to_stub:   whether to save the model to a stub (stub_path).
        :param stub_path:       the stub_path to load the model, or save the model.
        """
        # Storing the time_series and model_name:
        self.time_series = time_series
        self.model_name = model_name

        # Storing information about reading from and writing to stubs:
        self.read_from_stub = read_from_stub
        self.write_to_stub = write_to_stub
        self.stub_path = stub_path

        # Creating a list of features (as column names) for the dataset:
        self.features = time_series.features + time_series.lags
        # Target to predict (as a column name):
        self.target = time_series.value_name

        # Storing information about the model:
        self.model = None                           # the actual Model object for this instance
        self.model_kwargs = None                    # storing any special arguments used
        self.is_trained = False
        self.model_validation_loss = None           # validation loss
        self.model_validation_dataframe = None      # df_merged from validation loss

    def model_create(self):
        """
        Initializes this model's Model and stores it in self.model. Reads from stub if applicable.

        :return:    None.
        """
        if self.model_name == "XGBoostCV_v1":
            self.model = XGBoostCV_v1(self.time_series)
        elif self.model_name == "XGBoostTTV_v1":
            self.model = XGBoostTTV_v1(self.time_series)
        else:
            raise ValueError("Model not found. Check whether model_name is correct.")

        return

    def model_train(self):
        """
        Trains the model stored in self.model. Stores the trained model back in self.model.
        Requires that self.model_create() has been run.

        :return:        None.
        """
        self.is_trained = True
        return self.model.train()

    def model_predict(self, custom_df=None):
        """
        Predicts using the Model's predict() method. Requires that self.model_train() has been run.

        :param custom_df:       predicting on a custom DataFrame (optional).
        :return:                None.
        """
        return self.model.predict(custom_df)

    def model_get_validation_loss(self, loss_function="mean_squared_error", verbose=True):
        """
        Gets the validation loss of the model on the TimeSeries data of this instance. Saves the information in
        the object.

        :param loss_function:       the loss metric to use.
        :param verbose:             prints debugging information.
        :return:                    tuple[float (loss), pd.DataFrame (validation dataframe)]
        """
        self.model_validation_loss, self.model_validation_dataframe = validation_loss(self.model,
                                                                                      loss_function=loss_function,
                                                                                      verbose=verbose)
        return self.model_validation_loss, self.model_validation_dataframe

    def model_get_dict(self):
        """
        Creates a dictionary containing information about the MasterModel:
            {"model": Model, "hyperparameters": ..., "validation_loss": ..., "validation_df": ...,}

        :return:        dict() of MasterModel information.
        """
        model_dict = dict()
        model_dict["model"] = self.model
        model_dict["hyperparameters"] = self.model_kwargs
        model_dict["validation_loss"] = self.model_validation_loss
        model_dict["validation_df"] = self.model_validation_dataframe
        return model_dict
