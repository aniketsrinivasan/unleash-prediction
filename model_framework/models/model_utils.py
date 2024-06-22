import matplotlib.pyplot as plt
from sklearn.metrics import *
import numpy as np


def validation_loss(model, loss_function="mean_squared_error", verbose=True):
    """
    Consumes a Model, a loss_function, and runs the Model.predict() method on the
    TimeSeries validation split (TimeSeries.df_split_valid).

    Calculates loss, and creates a merged DataFrame containing three columns:
        DateTime, Target, and "Prediction_{Model.model_name}".

    :param model:           Model to be evaluated.
    :param loss_function:   loss function metric to use.
    :param verbose:         prints debugging information.
    :return:                tuple[float (loss), pd.DataFrame (df_merged)]
    """
    if verbose:
        print(f"Evaluating Model {model.model_name} on loss metric {loss_function}...")

    # Extracting the dataset to predict on:
    #   If Model is an XGBoost model, then the .predict() method takes in the
    #   the empty validation DataFrame and fills Target values based on features.
    if model.model_name in ("XGBoostCV_v1", "XGBoostTTV_v1"):
        df_prediction = model.time_series.df_split_valid.fillna(0).copy()

    #   If the Model is an LSTM model, then the .predict() method takes in the
    #   last "n" known sequential Target values, and predicts a set of future values.
    elif model.model_name in ("TorchLSTM_v1", "TorchLSTM_v2"):
        # Note: the last "n" known sequential Target values before TimeSeries.df_split_valid
        #       is end of TimeSeries.df_split_test.
        #       If no test split exists (so df_split_test is None), then use the
        #       train split (df_split_train_last_n) instead.
        if model.time_series.df_split_test_last_n is not None:
            if verbose:
                print(f"Running predictions for Model {model.model_name} on last 'n' of TEST set...")
            df_prediction = model.time_series.df_split_test_last_n.fillna(0).copy()
        else:
            print(f"Running predictions for Model {model.model_name} on last 'n' of TRAIN set...")
            df_prediction = model.time_series.df_split_train_last_n.fillna(0).copy()

    #   Otherwise this function has not been implemented for this Model variant:
    else:
        raise NotImplementedError(f"Invalid Model name {model.model_name}. "
                                  f"Check whether validation_loss() needs modification.")

    # Running the Model.predict() method:
    #   note: Model.predict() always takes a DataFrame and returns an array of predictions.
    #         The only thing that changes is our choice of input df_prediction, determined above.
    future_prediction = model.predict(custom_df=df_prediction,
                                      datetime_name=model.time_series.datetime_name,
                                      value_name=model.time_series.value_name)
    if verbose:
        print(f"Predictions created with size {future_prediction.shape}.")

    # Getting the validation dataset (the labels) to calculate loss:
    df_validation = model.time_series.df_split_valid.copy()

    # We check whether there are enough prediction datapoints to calculate loss:
    if df_validation.shape[0] > future_prediction.shape[0]:
        print(f"SoftWarn: Not enough information to calculate validation loss. \n"
              f"  Size of validation: {df_validation.shape[0]}. \n"
              f"  Size of prediction: {future_prediction.shape[0]}. ")
        # Setting all other (unaffected) values to zero:
        temp_padding_array = np.array([[0] for _ in range(df_validation.shape[0] - future_prediction.shape[0])])
        future_prediction = np.concatenate((future_prediction, temp_padding_array))
        print(f"Filled all missing values to (0) to form predictions of length {future_prediction.shape[0]}.")
    else:
        # We take only as many entries as we need (and we are guaranteed to have these many):
        future_prediction = future_prediction[:df_validation.shape[0]]

    # Calculating loss based on the provided loss function:
    if loss_function == "mean_squared_error":
        loss = mean_squared_error(df_validation[model.time_series.value_name], future_prediction)
    else:
        print(f"Invalid loss function ({loss_function}). Defaulting to mean_squared_error.")
        loss = mean_squared_error(df_validation[model.time_series.value_name], future_prediction)

    # Printing loss (if verbose):
    if verbose:
        print(f"Validation loss ({loss_function}):  {loss:.5f}")

    # Creating a new (merged) DataFrame containing prediction and label values:
    df_merged = df_validation[[model.time_series.value_name, model.time_series.datetime_name]].copy()
    df_merged[f"Prediction_{model.model_name}"] = future_prediction

    # Setting the index of this DataFrame to be the DateTime column (for plotting):
    df_merged.set_index(model.time_series.datetime_name, inplace=True)

    # Printing and plotting loss information if verbose:
    if verbose:
        print(loss)
        df_merged.plot()
        plt.ylim(bottom=0)
        plt.show()

    return loss, df_merged
