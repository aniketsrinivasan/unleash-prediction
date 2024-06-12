import pandas as pd


# data_read_csv(filename, verbose=False) reads a given .csv file and returns a variable containing
#   a pd.DataFrame of that file.
def data_read_csv(path: str, verbose=True) -> pd.DataFrame:
    """
    Reads data from a .csv file and returns a pd.DataFrame of that file.

    :param path:        path to the .csv file.
    :param verbose:     prints debugging information.
    :return:            pd.DataFrame.
    """

    if verbose:
        print(f'Reading file from {path}.')
    file = pd.read_csv(path)
    return file


# data_datetime_conversion(dataframe, datetime_name, datetime_format, verbose) converts the datetime column
#   of a dataframe to the DateTime format understood by Pandas.
# Saves in the new pd.DataFrame under the attribute "datetime_name".
def data_datetime_conversion(dataframe: pd.DataFrame, datetime_name: str,
                             datetime_format: str, verbose=True) -> pd.DataFrame:
    """
    Converts the date/time column of a dataframe into the DateTime format (understood by Pandas).

    :param dataframe:           dataframe to sort.
    :param datetime_name:       name of the date/time column in dataframe.
    :param datetime_format:     format of the date/time (e.g. "%Y-%m-%d %H:%M:%S").
    :param verbose:             prints debugging information.
    :return:                    pd.DataFrame.
    """

    if verbose:
        print(f'Converting {datetime_name} in {dataframe} to DateTime format.')
    dataframe_new = dataframe.copy()
    dataframe_new[datetime_name] = pd.to_datetime(dataframe_new[datetime_name], format=datetime_format)
    return dataframe_new


# data_datetime_sort(dataframe, datetime_name, verbose) consumes a dataframe and sorts it by datetime.
def data_datetime_sort(dataframe: pd.DataFrame, datetime_name: str, verbose=True) -> pd.DataFrame:
    """
    Consumes a dataframe and sorts by datetime_name.

    :param dataframe:           the dataframe to sort.
    :param datetime_name:       name of the DateTime column in dataframe.
    :param verbose:             prints debugging information.
    :return:                    pd.DataFrame (sorted by datetime_name).
    """

    if verbose:
        print(f'Sorting by {datetime_name} in {dataframe}.')
    dataframe_new = dataframe.copy()
    dataframe_new = dataframe_new.sort_values([datetime_name], ascending=[True])
    return dataframe_new


# data_datetime_create_features(dataframe, datetime_name) creates a set of features in the given pd.DataFrame.
#   Requires that dataframe is sorted by DateTime.
#   Returns a tuple:    modified pd.DataFrame, dictionary containing
def data_datetime_create_features(dataframe: pd.DataFrame, datetime_name: str,
                                  hours=True,
                                  days_of_week=True,
                                  weeks=True,
                                  days_of_month=False,
                                  months=False,
                                  verbose=True) -> tuple[pd.DataFrame, dict]:
    """
    Creates a set of time-series features (as columns) in the given pd.DataFrame.
    Returns a modified dataframe, and a dictionary of features added in the following format:
        {param: name_in_dataframe, ...}


    :param dataframe:           dataframe to add time-series features to.
    :param datetime_name:       name of the DateTime column in dataframe.
    :param hours:               add the "hour" feature.
    :param days_of_week:        add the "day of the week" feature.
    :param weeks:               add the "week of the year" feature.
    :param days_of_month:       add the "day of the month" feature.
    :param months:              add the "month of the year" feature.
    :param verbose:             prints debugging information.
    :return:                    tuple of pd.DataFrame, dictionary of features.
    """

    if verbose:
        print(f'Creating DateTime features.')
    dataframe_new = dataframe.copy()
    # Initializing dictionary to keep track of created column names:
    column_dict = dict()

    # Days since start of the dataset (added by default):
    #   requires the dataframe to be sorted by date already.
    dataframe_new['days_since_start'] = (dataframe_new[datetime_name]
                                         - dataframe_new[datetime_name].iloc[0]).dt.days.astype(int)
    column_dict['days_since_start'] = "days_since_start"

    # Hours (will almost always be used):
    if hours:
        dataframe_new["hour_of_day"] = dataframe_new[datetime_name].dt.strftime("%H").astype(int)
        column_dict["hours"] = "hour_of_day"

    # Weeks and days of week:
    if days_of_week:
        dataframe_new["day_of_week"] = dataframe_new[datetime_name].dt.isocalendar().day.astype(int)
        column_dict["days_of_week"] = "day_of_week"
    if weeks:
        dataframe_new["week_of_year"] = dataframe_new[datetime_name].dt.isocalendar().week.astype(int)
        column_dict["weeks"] = "week_of_year"

    # Months and days of month:
    if days_of_month:
        dataframe_new["day_of_month"] = dataframe_new[datetime_name].dt.strftime("%d").astype(int)
        column_dict["days_of_month"] = "day_of_month"
    if months:
        dataframe_new["month_of_year"] = dataframe_new[datetime_name].dt.strftime("%m").astype(int)
        column_dict["months"] = "month_of_year"

    return dataframe_new, column_dict


# data_create_lags(dataframe, value_name, lag_base, lag_multiplies) creates a lagged column of value_name
#   using lag_base (representing the "unit" to lag by) and lag_multiples (a list of how many units each
#   should lag by).
# Returns a modified pd.DataFrame and the minimum lag quantity (in entries) used.
def data_create_lags(dataframe: pd.DataFrame, value_name: str,
                     lag_base: int, lag_multiples: list[int], lag_label="",
                     verbose=True) -> tuple[pd.DataFrame, int, list]:
    """
    Creates lagged column(s) of value_name using lag_base (representing the "unit" to lag by) and
    lag_multiples (a list of how many units each column should be lagged by). Unfilled values are NaN.

    :param dataframe:       dataframe to create lags using.
    :param value_name:      name of column to lag.
    :param lag_base:        integer "base" unit to lag by (number of entries).
    :param lag_multiples:   list of multiples to lag by.
    :param lag_label:       the label for lag_base (e.g "week").
    :param verbose:         prints debugging information.
    :return:                tuple of pd.DataFrame, minimum lag (number of entries), list of lags (strings).
    """

    if verbose:
        print("Creating lags in dataset...")

    # Creating a list of the column names for all the lag columns created:
    lags = [f"lag_{multiple}{lag_label}" for multiple in lag_multiples]

    max_lag = max(lag_multiples) * lag_base
    if max_lag > dataframe.shape[0]:
        raise IndexError("Lags exceed dataset size.")
    dataframe_new = dataframe.copy()
    for lag_multiplier in lag_multiples:
        dataframe_new[f"lag_{lag_multiplier}{lag_label}"] = dataframe_new[value_name].shift(lag_multiplier * lag_base)
    return dataframe_new, min(lag_multiples)*lag_base, lags


# data_split_features_labels(dataframe, value_name, verbose) splits the dataset into:
#   training set:   contains everything except column value_name (features)
#   labels:         contains only value_name (labels)
# and returns both as a tuple.
def data_split_features_labels(dataframe: pd.DataFrame, value_name: str, verbose=True) -> tuple:
    """
    Splits the dataframe into two parts:
        training set:   (training) contains everything except the column value_name;
        labels:         (labels) contains only column value_name.

    :param dataframe:       dataframe to split.
    :param value_name:      the column to split using.
    :param verbose:         prints debugging information.
    :return:                tuple of pd.DataFrame in the order (training, labels)
    """

    if verbose:
        print("Splitting dataset into training and testing.")
    X_data = dataframe.drop([value_name], axis=1).copy()
    y_data = dataframe[value_name].copy()
    return X_data, y_data


# data_split_train_test_valid(dataframe, ratio, verbose) takes pd.DataFrame, a list of floats representing
#   ratios (for train/test/valid respectively), and returns three pd.DataFrames (split accordingly).
def data_split_train_test_valid(dataframe: pd.DataFrame, ratio: list[float], verbose=True):
    """
    Splits a dataframe into three parts (training, testing, validation) based on ratios.

    :param dataframe:       pd.DataFrame to split.
    :param ratio:           a list of floats (adding to 1) for split ratios (train, test, valid).
    :param verbose:         prints progress to console.
    :return:                triple of pd.DataFrame split in the order (train, test, valid).
    """

    if sum(ratio) != 1:
        raise ValueError("Ratios must sum to 1.")
    if verbose:
        print("Splitting dataset into training, testing and validation.")
    dataset_len = dataframe.shape[0]
    train_size = int(dataset_len * ratio[0])
    test_size = int(dataset_len * ratio[1])
    valid_size = int(dataset_len * ratio[2])

    data_train = dataframe.iloc[:train_size]
    data_test = dataframe.iloc[train_size:train_size+test_size]
    data_valid = dataframe.iloc[-valid_size:]

    return data_train, data_test, data_valid


# data_create_future(dataframe, datetime_name, value_name, window_size, step) consumes a pd.DataFrame,
#   a window_size and a step (defaulted to "1h").
#   It returns a new pd.DataFrame with window_size new entries, each with a DateTime index stepping by step.
# The returned pd.DataFrame has Datetime titled datetime_name, and empty values titled value_name.
# If append=True, then a copy of dataframe is appended to the future set.
# Requires:     dataframe is sorted by datetime_name, which is in DateTime format
def data_create_future(dataframe, datetime_name: str, value_name: str,
                       window_size: int, step="1h", append=True, verbose=True):
    """
    Consumes a pd.DataFrame, and creates a "future" DataFrame of (only) the DateTime and Target columns.

    :param dataframe:       input dataframe.
    :param datetime_name:   name of DateTime column in dataframe.
    :param value_name:      name of Target column in dataframe.
    :param window_size:     number of entries to create in the future.
    :param step:            the step size to create future DateTime column.
    :param append:          whether to append to the original dataframe.
    :param verbose:         prints progress to console.
    :return:                pd.DataFrame for the future.
    """

    if verbose: print(f"Creating future of size {window_size} with step {step}.")
    # Copying the existing dataframe and finding the last timestamp:
    dataframe_new = dataframe.copy()
    last_date = dataframe_new[datetime_name].iloc[-1]

    # Creating a new dataframe starting at the last date, stepping by step, with window_size size:
    future = pd.date_range(last_date, periods=window_size, freq=step)
    future_dataframe = pd.DataFrame({datetime_name: future})
    # Initializing the value_name to 0:
    future_dataframe[value_name] = 0

    if append:
        try:
            dataframe_concat = pd.concat([dataframe, future_dataframe])
            return dataframe_concat
        except Exception as e:
            print(f"Exception: {e}. Returning non-concatenated dataframe.")

    return future_dataframe
