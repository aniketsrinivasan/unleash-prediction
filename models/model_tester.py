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


class ModelTester:
    def __init__(self, kwargs_timeseries_init, kwargs_timeseries_prepare):
        self.time_series = TimeSeries(**kwargs_timeseries_init)
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
                this_model.model_get_validation_loss(loss_function=loss_function, verbose=verbose)
            model_dict[model] = this_model

        # Saving model_dict in the instance:
        self.model_dict = model_dict
        return

    def run_validation(self):
        """
        Gets validation information on all models stored in self.model_dict.
            If train_if_needed is set to True, then untrained models in self.model_dict will be trained.
            If validate_if_needed is set to True, then models without a validation_loss will be validated.

        :return:        None.
        """
        pass

    def get_validation_losses(self):
        pass
