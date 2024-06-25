import pandas as pd
from ..LSTM import TorchLSTM_v1, TorchLSTM_v2
from ..xgboost import XGBoostTTV_v1, XGBoostCV_v1
from utils import TimeSeries
from ...models import MasterModel


def average_mean(predictions: pd.DataFrame, verbose=False):
    pass


def average_linear_split(prediction1, prediction2, ceil_floor_values: tuple,
                         ceil_length: int, linear_ratio: float, verbose=False):
    pass


# Using Dynamic Averaging methods between XGBoostTTV_v1 and TorchLSTM_v2 to come up with predictions
#   for datasets containing high correlation with wave-like behaviour.


class HybridAveraging_v1:
    def __init__(self, time_series: TimeSeries, models_init, verbose=True):
        """
        Initializes a HybridAveraging model for predictions, which is a multi-model architecture
        that generates multiple predictions (using models in models_init), and then combines them
        using various averaging methods.

        :param time_series:     the TimeSeries to predict using.
        :param models_init:     model initialization dictionary.
        :param verbose:         prints debugging information.
        """
        self.verbose = verbose

        # Initializing and preparing the TimeSeries:
        self.time_series = time_series

        # Saving Model initialization information:
        self.models_init = models_init

        # Initializing a Model dictionary that stores the provided Models:
        model_dict = {}
        for model in self.models_init:
            model_dict[model] = MasterModel(time_series=time_series, model_name=model,
                                            **models_init[model])
        self.model_dict = model_dict

    def train(self):
        if self.verbose:
            print(f"Training Models for HybridAveraging...")
        for master_model in list(self.model_dict.values()):
            if not master_model.is_trained:
                if self.verbose:
                    print(f"Training model {master_model.model_name}...")
                master_model.model_train()
            elif self.verbose:
                print(f"Model {master_model.model_name} already trained. Skipping.")
        return

    def predict(self):
        pass
