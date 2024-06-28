import pandas as pd
from macros import TorchLSTM_v2_LOOKBACK
from utils import TimeSeries
from ...models import MasterModel


def create_linear_split(lookback: int, start_end_values: tuple, smooth_window_ratio: float):
    """
    Creates two lists of floats, both adding to 1. The first list follows the above pattern,
    such that the values in the list begin at start_end[0], and end at start_end[1].
    The second list is (1-first_list).

    Both lists are of length lookback.

    :param lookback:            the length of the produced list.
    :param start_end_values:    a tuple of (start_value, end_value) for calculating averages.
    :param smooth_window_ratio: the ratio (of lookback) for the smoothing window.
    :return:                    a tuple of float lists (scheduler_list, complement_list).
    """
    # Initializing scheduler_list:
    scheduler_list = [0 for _ in range(lookback)]
    smooth_start_idx = int(lookback - (smooth_window_ratio * lookback))
    smooth_slope = (start_end_values[1] - start_end_values[0]) / (lookback - smooth_start_idx)
    for i in range(lookback):
        if i < smooth_start_idx:
            scheduler_list[i] = start_end_values[0]
        else:
            scheduler_list[i] = start_end_values[0] + smooth_slope * (i - smooth_start_idx)

    complement_list = [(1 - x) for x in scheduler_list]
    return scheduler_list, complement_list


def average_linear_split(predictions: pd.DataFrame, isolate: str, ceil_floor_values: tuple,
                         smooth_ratio: float, verbose=False):
    isolate_df = predictions[isolate]
    complement_df = predictions.copy().drop(columns=[isolate]).mean(axis=1)

    # Identifying the meaningful future window (this is the window to which TorchLSTM_v2 can
    #   predict without excess divergence):
    meaningful_future_window = min(TorchLSTM_v2_LOOKBACK, isolate_df.shape[0])
    # Creating a linear split scheduler:
    scheduler_list, complement_list = create_linear_split(meaningful_future_window, ceil_floor_values,
                                                          smooth_ratio)

    averages = list(complement_df)
    # Taking averages based on scheduler:
    for i in range(meaningful_future_window):
        this_average = scheduler_list[i] * isolate_df.iloc[i] + complement_list[i] * \
                       complement_df.iloc[i]
        averages[i] = this_average

    return averages


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
            this_model = MasterModel(time_series=time_series, model_name=model,
                                     **models_init[model])
            this_model.model_create()
            model_dict[model] = this_model
        self.model_dict = model_dict

        # Storing predictions (initializing a predictions DataFrame):
        self.combined_prediction_df = self.time_series.df_future_only[[self.time_series.datetime_name]].copy()
        self.combined_prediction_df.set_index(self.time_series.datetime_name, inplace=True)
        # Name of the averaged predictions column in the DataFrame:
        self.predictions_name = None

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

    def incremental_train(self):
        if self.verbose:
            print(f"Incrementally training Models for HybridAveraging...")
        for master_model in list(self.model_dict.values()):
            if self.verbose:
                print(f"Running incremental training on {master_model.model_name}.")
            master_model.model_incremental_train()
        return

    def predict(self, averaging="average_mean", kwargs_averaging=None):
        for master_model in list(self.model_dict.values()):
            if not master_model.is_trained:
                print(f"! Critical Warning: Model {master_model.model_name} is not trained. "
                      f"Running predictions regardless. !")
            # Adding this prediction to the combined_prediction_df:
            self.combined_prediction_df[f"Prediction_{master_model.model_name}"] = (
                master_model.model_predict())

        if averaging == "average_mean":
            self.combined_prediction_df[averaging] = self.combined_prediction_df.mean(axis=1)
        elif averaging == "average_linear_split":
            self.combined_prediction_df[averaging] = average_linear_split(self.combined_prediction_df,
                                                                          **kwargs_averaging)
        return self.combined_prediction_df, averaging

