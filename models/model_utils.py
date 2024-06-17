import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


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
