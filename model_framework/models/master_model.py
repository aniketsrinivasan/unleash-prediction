from .xgboost import *
from .model_utils import validation_loss
from utils import TimeSeries
from .LSTM import TorchLSTM_v1


class MasterModel:
    def __init__(self, time_series: TimeSeries, model_name: str,
                 read_from_stub=None, write_to_stub=None, is_trained=True):
        # Initialize TimeSeries object:
        self.time_series = time_series
        self.verbose = time_series.verbose

        # Instance information:
        self.model_name = model_name
        # Saving read/write information:
        self.read_from_stub = read_from_stub
        self.write_to_stub = write_to_stub
        # If read_from_stub is provided, then is_trained can be used:
        self.is_trained = is_trained if (read_from_stub is not None) else False

        # Storing information about the model:
        self.model = None               # the actual Model object for this instance
        self.model_kwargs = None        # storing any special arguments used (for saving later)

        # Storing validation information about the model:
        self.model_validation_loss = None
        self.model_validation_df = None         # DataFrame containing validation labels and predictions

    def model_create(self):
        """
        Initializes this MasterModel's Model (regressor) and stores it in self.model.
        Reads an existing model from self.read_from_stub if provided. Otherwise, it creates a new
        instance of the model.

        :return:    None.
        """
        if self.verbose:
            print(f"Initializing model {self.model_name}...")
        # Initializing models:
        if self.model_name == "XGBoostCV_v1":
            self.model = XGBoostCV_v1(time_series=self.time_series,
                                      read_from_stub=self.read_from_stub,
                                      write_to_stub=self.write_to_stub)
        elif self.model_name == "XGBoostTTV_v1":
            self.model = XGBoostTTV_v1(time_series=self.time_series,
                                       read_from_stub=self.read_from_stub,
                                       write_to_stub=self.write_to_stub)
        elif self.model_name == "TorchLSTM_v1":
            self.model = TorchLSTM_v1(time_series=self.time_series,
                                      read_from_stub=self.read_from_stub,
                                      write_to_stub=self.write_to_stub)
        else:
            raise ValueError(f"Model {self.model_name} not found. Unable to initialize.")

        return

    def model_train(self):
        """
        Trains the model stored in self.model. Requires that self.create() has been run.
        If self.is_trained is True, then this function is passed and does NOT override the model.

        :return:    None.
        """
        if self.verbose:
            print(f"Training model {self.model_name}...")
        if self.is_trained:
            print(f"Model {self.model_name} has already been trained. Not continuing training.")
            return
        # Set the model as trained (to avoid re-training):
        self.is_trained = True
        return self.model.train()

    def model_predict(self, custom_df=None, datetime_name=None, value_name=None):
        if self.verbose:
            print(f"Running predictions using model {self.model_name}. "
                  f"Custom DataFrame: {True if custom_df is not None else False}.")
        if not self.is_trained:
            print(f"SoftWarn: Model ({self.model_name}) is not trained. "
                  f"Use MasterModel.train() before making predictions.")
        return self.model.predict(custom_df=custom_df, datetime_name=datetime_name, value_name=value_name)

    def model_run_validation(self, loss_function="mean_squared_error"):
        """
        Runs the model_utils.validation_loss() function on the TimeSeries of this MasterModel.
        Saves the information in self.model_validation_loss and self.model_validation_df.

        The model_utils.validation_loss() function runs the model's forecasting (prediction)
        as well as calculates the loss.

        :return:    tuple[float (loss), pd.DataFrame (validation DataFrame)]
        """
        self.model_validation_loss, self.model_validation_df = validation_loss(model=self.model,
                                                                               loss_function=loss_function,
                                                                               verbose=self.verbose)
        return self.model_validation_loss, self.model_validation_df

    def get_dict(self):
        """
                Creates a dictionary containing information about the MasterModel:
                    {"model": Model, "hyperparameters": ..., "validation_loss": ..., "validation_df": ...,}

                :return:        dict() of MasterModel information.
                """
        if self.verbose:
            print(f"Creating MasterModel dictionary...")
        model_dict = dict()
        model_dict["model"] = self.model
        model_dict["hyperparameters"] = self.model_kwargs
        model_dict["validation_loss"] = self.model_validation_loss
        model_dict["validation_df"] = self.model_validation_df
        if self.verbose:
            print(f"MasterModel dictionary created: \n"
                  f"    {model_dict}.")
        return model_dict
