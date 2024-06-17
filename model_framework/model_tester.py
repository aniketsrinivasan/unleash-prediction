from .models import MasterModel
from utils import TimeSeries
# Idea: test and compare various model_framework easily.

# functions to implement:
#   create a class called ModelTester (maybe subclass of TimeSeries?)
#       store all initiated model_framework and their results in a dictionary? {model_name: results, ...}
#       __init__(TimeSeries):       initialization using time series

#       train_models(model_framework=ModelTester.model_list):
#           trains model_framework and stores objects somewhere (maybe something like ModelTester.trained_models)
#           defaults to training all of them.

#       test_models(model_framework=ModelTester.model_list, window_base, window_multiples, train=True):
#           tests model_framework in "model_framework" (defaults to list of all). trains if not already trained if train==True.
#           trained model_framework are taken from wherever train_models() saves them.
#           testing occurs on validation dataset of TimeSeries.
#           size of the window is determined by window_base and window_multiples (capped at validation size)
#           creates (and returns) new DataFrame with columns for all model predictions

#   find a way to save hyperparameters automatically when running model_framework.
#       perhaps save in a dictionary file somewhere (load "write", modify, close) each time
#       maybe worth creating a class for saved results (in a struct-style) and saving this instead?


class ModelTester:
    def __init__(self, kwargs_timeseries_init, kwargs_timeseries_prepare,
                 kwargs_features=None, kwargs_lags=None):
        self.time_series = TimeSeries(**kwargs_timeseries_init)
        self.time_series.prepare_from_scratch(**kwargs_timeseries_prepare,
                                              kwargs_features=kwargs_features,
                                              kwargs_lags=kwargs_lags)

        self.kwargs_timeseries_init = kwargs_timeseries_init        # saving args used for TimeSeries init
        self.kwargs_timeseries_prepare = kwargs_timeseries_prepare  # saving args used for TimeSeries prepare

        # Storing a list of model_framework:
        self.model_list = ["XGBoostCV_v1", "XGBoostTTV_v1"]     # automate this later

        # Initializing an empty dictionary to save trained model_framework and results:
        self.model_dict = None
        # When actually creating a savefile for ModelTester, make sure to also save all information
        #   needed to initialize the TimeSeries.
        # Do not save pre-trained model_framework. Only save the TimeSeries information and hyperparameters used
        #   (this is for memory efficiency).

    def create_model_dict(self, train=False, validate=False, loss_function="mean_squared_error",
                          read_from_stubs=False, write_to_stubs=False, stub_dir=None, verbose=True):
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

    def run_training(self, override=False, verbose=True):
        """
        Runs training on all model_framework stored in self.model_dict.
            If override=True, then any existing trained model is re-trained.

        :param override:        overrides any existing training information.
        :param verbose:         prints debugging information.
        :return:                None.
        """
        if verbose:
            print(f"Running training on all model_framework with override={override}:")
        for master_model in list(self.model_dict.values()):
            if (override) or (not master_model.is_trained):
                if verbose:
                    print(f"Training model {master_model.model_name}...")
                master_model.model_train()
            elif verbose:
                print(f"Model {master_model.model_name} already trained. Skipping.")
        return

    def run_validation(self, loss_function="mean_squared_error", override=False, verbose=False):
        """
        Gets validation information on all model_framework stored in self.model_dict.
            If override=True, any existing validation information is re-validated.

        :param: override:       overrides existing validation information.
        :return:                None.
        """
        if verbose:
            print(f"Running validation on all model_framework with override={override}:")
        for master_model in list(self.model_dict.values()):
            if (override) or (master_model.model_validation_loss is None):
                if verbose:
                    print(f"Validating model {master_model.model_name}...")
                master_model.model_get_validation_loss(loss_function, verbose=verbose)
            elif verbose:
                print(f"Model {master_model.model_name} already validated. Skipping.")
        return

    def get_validation_losses(self):
        """
        Prints and returns a dictionary of validation losses from self.model_dict.

        :return:        dict() of validation losses (model_name: validation_loss).
        """
        losses = dict()
        for master_model in list(self.model_dict.values()):
            losses[master_model.model_name] = master_model.model_validation_loss
        print(losses)
        return losses
