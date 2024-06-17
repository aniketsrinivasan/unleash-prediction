from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from .xgboost import *
from .master_model import MasterModel
from utils import TimeSeries
# Idea: test and compare various models easily.

# functions to implement:
#   create a class called ModelTester (maybe subclass of TimeSeries?)
#       store all initiated models and their results in a dictionary? {model_name: results, ...}
#       __init__(TimeSeries):       initialization using time series

#       train_models(models=ModelTester.model_list):
#           trains models and stores objects somewhere (maybe something like ModelTester.trained_models)
#           defaults to training all of them.

#       test_models(models=ModelTester.model_list, window_base, window_multiples, train=True):
#           tests models in "models" (defaults to list of all). trains if not already trained if train==True.
#           trained models are taken from wherever train_models() saves them.
#           testing occurs on validation dataset of TimeSeries.
#           size of the window is determined by window_base and window_multiples (capped at validation size)
#           creates (and returns) new DataFrame with columns for all model predictions

#   find a way to save hyperparameters automatically when running models.
#       perhaps save in a dictionary file somewhere (load "write", modify, close) each time
#       maybe worth creating a class for saved results (in a struct-style) and saving this instead?


def validation_loss(model, loss_function="mean_squared_error", verbose=True):
    """
    Consumes a Model, a loss_function and runs the Model.predict() method on the TimeSeries validation
    split. It calculates the loss on this data, and returns the loss and merged DataFrame (containing
    three columns: DateTime, Target and "Prediction_{model_name}".

    :param model:               the Model to be evaluated.
    :param loss_function:       the loss function metric to use.
    :param verbose:             prints debugging information.
    :return:                    tuple[float (loss), DataFrame (df_merged)]
    """
    # Extracting the dataset to predict on:
    #   this is the validation set pre-prediction (with only feature columns)
    df_prediction = model.time_series.df_split_valid.fillna(0).copy()
    # Running predictions on this dataset:
    future_prediction = model.predict(df_prediction)
    # Getting the validation dataset (labels)
    df_validation = model.time_series.df_split_valid.copy()

    # We check whether there are enough validation datapoints to calculate loss:
    if df_validation.shape[0] < future_prediction.shape[0]:
        raise Exception(f"Not enough information to calculate validation loss.")

    # We take only as many entries as we need (and we are guaranteed to have these many):
    df_validation = df_validation.iloc[:future_prediction.shape[0]]

    # Calculating loss based on the provided loss function:
    if loss_function == "mean_squared_error":
        loss = mean_squared_error(df_validation[model.time_series.value_name], future_prediction)
    else:
        print(f"Invalid loss function ({loss_function}). Defaulting to mean_squared_error.")
        loss = mean_squared_error(df_validation[model.time_series.value_name], future_prediction)

    if model.time_series.verbose:
        print(f"Validation loss (mean_squared):    {loss:.4f}")

    # Creating a new (merged) DataFrame:
    df_merged = df_validation[[model.time_series.value_name, model.time_series.datetime_name]].copy()
    df_merged[f"Prediction_{model.model_name}"] = future_prediction
    # Setting index of this DataFrame to be the DateTime column:
    df_merged.set_index(model.time_series.datetime_name, inplace=True)

    if verbose:
        print(loss)
        # Plotting the merged dataset:
        df_merged.plot()
        plt.show()

    return loss, df_merged


class ModelTester(TimeSeries):
    def __init__(self, kwargs_timeseries_init, kwargs_timeseries_prepare):
        self.time_series = super().__init__(**kwargs_timeseries_init)
        self.time_series.prepare_from_scratch(**kwargs_timeseries_prepare)

        self.kwargs_timeseries_init = kwargs_timeseries_init        # saving args used for TimeSeries init
        self.kwargs_timeseries_prepare = kwargs_timeseries_prepare  # saving args used for TimeSeries prepare

        # Storing a list of models:
        self.model_list = ["XGBoostCV_v1", "XGBoostTTV_v1"]     # automate this later

        # Initializing an empty dictionary to save trained models and results:
        self.model_dict = None
        # When actually creating a savefile for ModelTester, make sure to also save all information
        #   needed to initialize the TimeSeries.
        # Do not save pre-trained models. Only save the TimeSeries information and hyperparameters used
        #   (this is for memory efficiency).

    def create_model_dict(self, train=False, validate=False, loss_function="mean_squared_error", verbose=True):
        # Fully initializing model_dict with the following structure:
        #   {model_name: MasterModel}
        if verbose:
            print(f"Initializing model dictionary for ModelTester...")
        model_dict = dict()

        for model in self.model_list:
            if verbose:
                print(f"Creating model {model}:")
            # Initializing this model (using MasterModel):
            this_model = MasterModel(time_series=self.time_series, model_name=model)
            this_model.model_create()
            # If train is True, then we train the model:
            if train:
                if verbose:
                    print(f"Training model {model}...")
                this_model.model_train()
            # If validate is True, then we run validation with verbose=False:
            if validate:
                if verbose:
                    print(f"Validating model {model}...")
                this_model.model_get_validation_loss(loss_function=loss_function, verbose=False)
            model_dict[model] = this_model

        # Saving model_dict in the instance:
        self.model_dict = model_dict
        return
