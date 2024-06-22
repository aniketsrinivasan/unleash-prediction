from .models import MasterModel
from utils import TimeSeries
import matplotlib.pyplot as plt
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
    def __init__(self, kwargs_timeseries_init, models_init, verbose=False):
        self.verbose = verbose
        self.time_series = TimeSeries(**kwargs_timeseries_init)
        self.time_series.prepare_from_scratch()

        self.kwargs_timeseries_init = kwargs_timeseries_init        # saving args used for TimeSeries init
        self.models_init = models_init                              # saving args used for Model init

        # Initializing an empty dictionary to save trained model_framework and results:
        self.model_dict = None
        # Initializing a DataFrame that contains each Model's validation prediction:
        self.combined_validation_df = None

    def create_model_dict(self, train=False, validate=False,
                          loss_function="mean_squared_error"):
        # Fully initializing model_dict with the following structure:
        #   {model_name: MasterModel}
        if self.verbose:
            print(f"Initializing model dictionary for ModelTester...")
        model_dict = dict()

        # Iterating over the keys in self.models_init (which are Model names):
        for model in self.models_init:
            if self.verbose:
                print(f"Creating model {model}:")
            # Initializing this model (using MasterModel):
            this_model = MasterModel(time_series=self.time_series, model_name=model,
                                     **self.models_init[model])
            this_model.model_create()
            # If train is True, then we train the model:
            if train:
                if self.verbose:
                    print(f"Training model {model}...")
                # This training only occurs if Model.is_trained is False:
                this_model.model_train()
            # If validate is True, then we run validation with verbose=False:
            if validate:
                if self.verbose:
                    print(f"Validating model {model}...")
                this_model.model_run_validation(loss_function=loss_function)
            model_dict[model] = this_model

        # Saving model_dict in the instance:
        self.model_dict = model_dict
        return

    def run_training(self, override=False):
        """
        Runs training on all model_framework stored in self.model_dict.
            If override=True, then any existing trained model is re-trained.

        :param override:        overrides any existing training information.
        :return:                None.
        """
        if self.verbose:
            print(f"Running training on all model_framework with override={override}:")
        for master_model in list(self.model_dict.values()):
            if override or (not master_model.is_trained):
                if self.verbose:
                    print(f"Training model {master_model.model_name}...")
                master_model.model_train()
            elif self.verbose:
                print(f"Model {master_model.model_name} already trained. Skipping.")
        return

    def run_validation(self, loss_function="mean_squared_error", override=False):
        """
        Gets validation information on all model_framework stored in self.model_dict.
            If override=True, any existing validation information is re-validated.

        :param: override:       overrides existing validation information.
        :return:                None.
        """
        if self.verbose:
            print(f"Running validation on all model_framework with override={override}:")
        for master_model in list(self.model_dict.values()):
            if override or (master_model.model_validation_loss is None):
                if self.verbose:
                    print(f"Validating model {master_model.model_name}...")
                master_model.model_run_validation(loss_function)
            elif self.verbose:
                print(f"Model {master_model.model_name} already validated. Skipping.")
        return

    def get_validation_losses(self):
        """
        Prints and returns a dictionary of validation losses from self.model_dict.

        :return:        dict() of validation losses (model_name: validation_loss).
        """
        losses = dict()
        for model in list(self.model_dict.values()):
            losses[model.model_name] = model.model_validation_loss
        print(losses)
        return losses

    def plot_validation_losses(self):
        # Extracting the MasterModel.model_validation_df for each MasterModel:
        combined_df = self.time_series.df_split_valid[[self.time_series.datetime_name,
                                                       self.time_series.value_name]].copy()
        combined_df.set_index(self.time_series.datetime_name, inplace=True)

        for model in list(self.model_dict.values()):
            combined_df[model.model_name] = model.model_validation_df[f"Prediction_{model.model_name}"]

        self.combined_validation_df = combined_df
        # Plotting:
        combined_df.plot()
        plt.ylim(bottom=0)
        plt.show()
        # Setting the mean value (for later):
        self.combined_validation_df["Mean_Prediction"] = self.combined_validation_df.mean(axis=1)
        return

    # Plotting the average prediction of all predictions on the validation set:
    def plot_validation_mean(self):
        if self.combined_validation_df is None:
            combined_df = self.time_series.df_split_valid[[self.time_series.datetime_name,
                                                           self.time_series.value_name]].copy()
            combined_df.set_index(self.time_series.datetime_name, inplace=True)
            for model in list(self.model_dict.values()):
                combined_df[model.model_name] = model.model_validation_df[f"Prediction_{model.model_name}"]

            combined_df["Mean_Prediction"] = combined_df.mean(axis=1)
            self.combined_validation_df = combined_df

        self.combined_validation_df[["Mean_Prediction", self.time_series.value_name]].plot()
        plt.ylim(bottom=0)
        plt.show()
        return

