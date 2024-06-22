import pandas as pd
import math
import holidays


# This module contains functions that use Pandas, Math and (Holidays) to modify data.
# These functions form the base implementation for the TimeSeries class, which manages
#   datasets comprehensively.


# data_read_csv() requires that "path" contains a .csv file. Not asserted.
def data_read_csv(path: str, verbose=True) -> pd.DataFrame:
    """
    Reads data from a .csv file and returns it as a pd.DataFrame.

    :param path:        path to the .csv file (absolute).
    :param verbose:     prints debugging information.
    :return:            pd.DataFrame (of file in path).
    """
    if verbose:
        print(f"Read file from {path} into a DataFrame.")
    return pd.read_csv(path)


def data_datetime_conversion(dataframe: pd.DataFrame, datetime_name: str,
                             datetime_format: str, verbose=True) -> pd.DataFrame:
    """
    Converts the DateTime column of a DataFrame into the pd.Datetime format.

    :param dataframe:           pd.DataFrame to sort.
    :param datetime_name:       name of the DateTime column in dataframe (as a string).
    :param datetime_format:     format of the DateTime column (e.g. "%Y-%m-%d %H:%M:%S") (as a string).
    :param verbose:             prints debugging information.
    :return:                    pd.DataFrame (with DateTime column as pd.Datetime values).
    """
    if verbose:
        print(f"Converting DateTime column {datetime_name} ({datetime_format}) to "
              f"pd.Datetime values.")
    if datetime_name not in dataframe.columns:
        raise IndexError(f"Could not find DateTime column {datetime_name}.")
    dataframe_new = dataframe.copy()
    dataframe_new[datetime_name] = pd.to_datetime(dataframe_new[datetime_name], format=datetime_format)
    return dataframe_new


def data_datetime_sort(dataframe: pd.DataFrame, datetime_name: str, verbose=True) -> pd.DataFrame:
    """
    Consumes a pd.DataFrame and sorts it by DateTime.

    :param dataframe:       pd.DataFrame to sort.
    :param datetime_name:   name of the DateTime column in dataframe (as a string).
    :param verbose:         prints debugging information.
    :return:                pd.DataFrame (sorted by DateTime).
    """
    if verbose:
        print(f"Sorting DataFrame by {datetime_name}.")
    if datetime_name not in dataframe.columns:
        raise KeyError(f"Could not find DateTime column {datetime_name}.")
    dataframe_new = dataframe.copy()
    dataframe_new = dataframe_new.sort_values([datetime_name], ascending=[True])
    return dataframe_new


# data_create_features() requires the pd.DataFrame dataframe to be sorted by DateTime.
#   Not asserted.
def data_create_features(dataframe: pd.DataFrame, datetime_name: str, value_name: str,
                         days_since_start=False,
                         hours=False,
                         days_of_week=False,
                         days_of_week_onehot=False,
                         weeks=False,
                         days_of_month=False,
                         months=False,
                         rolling_windows=None,
                         holidays_country=None,
                         holidays_province=None,
                         verbose=True) -> tuple[pd.DataFrame, dict]:
    """
    Creates a set of time-series features (as columns) in the given pd.DataFrame.
    Returns a modified dataframe, and a dictionary of features added in the following format:
        {param: name_in_dataframe, ...}


    :param dataframe:           dataframe to add time-series features to.
    :param datetime_name:       name of the DateTime column in dataframe.
    :param value_name:          name of the Target column in dataframe.
    :param days_since_start:    add the "days_since_start" feature (acts like an index count).
    :param hours:               add the "hour" feature.
    :param days_of_week:        add the "day of the week" feature (NOT one-hot encoded).
    :param days_of_week_onehot: add features for "days of the week" that are one-hot encoded.
    :param weeks:               add the "week of the year" feature.
    :param days_of_month:       add the "day of the month" feature.
    :param months:              add the "month of the year" feature.
    :param rolling_windows:     a list of integers representing rolling window entries.
    :param holidays_country:    the country code to add the public holidays of.
    :param holidays_province:   the province code to add the public holidays of.
    :param verbose:             prints debugging information.
    :return:                    tuple of pd.DataFrame, dictionary of features.
    """
    if verbose:
        print(f"Creating time-series features in the DataFrame.")
    dataframe_new = dataframe.copy()
    # Initializing dictionary to keep track of created column names:
    column_dict = dict()

    # Days since start of the dataset:
    if days_since_start:
        dataframe_new['days_since_start'] = (dataframe_new[datetime_name]
                                             - dataframe_new[datetime_name].iloc[0]).dt.days.astype(int)
        column_dict['days_since_start'] = "days_since_start"

    # Hours (interpreted as hour of the day):
    if hours:
        dataframe_new["hour_of_day"] = dataframe_new[datetime_name].dt.strftime("%H").astype(int)
        column_dict["hours"] = "hour_of_day"

    # Days of week (not one-hot encoded):
    if days_of_week:
        dataframe_new["day_of_week"] = dataframe_new[datetime_name].dt.isocalendar().day.astype(int)
        column_dict["days_of_week"] = "day_of_week"
    # Days of week (one-hot encoded):
    if days_of_week_onehot:
        raise NotImplementedError

    # Weeks of the year:
    if weeks:
        dataframe_new["week_of_year"] = dataframe_new[datetime_name].dt.isocalendar().week.astype(int)
        column_dict["weeks"] = "week_of_year"

    # Days of month:
    if days_of_month:
        dataframe_new["day_of_month"] = dataframe_new[datetime_name].dt.strftime("%d").astype(int)
        column_dict["days_of_month"] = "day_of_month"

    # Months of the year:
    if months:
        dataframe_new["month_of_year"] = dataframe_new[datetime_name].dt.strftime("%m").astype(int)
        column_dict["months"] = "month_of_year"

    # Creating rolling windows:
    if rolling_windows is not None:
        for n in rolling_windows:
            # Creating a column for this multiple of rolling mean:
            dataframe_new[f"rolling_{n}"] = dataframe_new[value_name].rolling(n).mean().fillna(0).astype(int)
            column_dict[f"rolling_{n}"] = f"rolling_{n}"

    # Holidays feature (one-hot encoded):
    if holidays_country is not None:
        all_datetimes = dataframe_new[datetime_name].to_list()
        is_holiday_list = []
        # If province is provided, search by province:
        if holidays_province is not None:
            for datetime in all_datetimes:
                is_holiday = holidays.country_holidays(holidays_country, prov=holidays_province).get(datetime)
                is_holiday_list.append(1 if is_holiday is not None else 0)
            dataframe_new[f"holiday_{holidays_country}_{holidays_province}"] = is_holiday_list
            column_dict[f"holiday"] = f"holiday_{holidays_country}_{holidays_province}"
        else:
            for datetime in all_datetimes:
                is_holiday = holidays.country_holidays(holidays_country).get(datetime)
                is_holiday_list.append(1 if is_holiday is not None else 0)
            dataframe_new[f"holiday_{holidays_country}"] = is_holiday_list
            column_dict["holiday"] = f"holiday_{holidays_country}"

    if verbose:
        print(f"Created time-series features for the DataFrame: \n"
              f"    {column_dict}")

    return dataframe_new, column_dict


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
    :return:                tuple of pd.DataFrame, minimum lag (number of entries), list of lag_names (strings).
    """
    if verbose:
        print(f"Creating lags in DataFrame...")

    # Creating a list of the column names for all the lag columns created:
    lag_names = [f"lag_{multiple}{lag_label}" for multiple in lag_multiples]

    if len(lag_multiples) > 0:
        max_lag = max(lag_multiples) * lag_base
        min_lag = min(lag_multiples) * lag_base
        if max_lag > dataframe.shape[0]:
            raise IndexError(f"Lag size ({max_lag}) exceeds dataset size ({dataframe.shape[0]}).")
    else:
        max_lag = min_lag = dataframe.shape[0]
    dataframe_new = dataframe.copy()
    for lag_multiplier in lag_multiples:
        dataframe_new[f"lag_{lag_multiplier}{lag_label}"] = dataframe_new[value_name].shift(
            lag_multiplier * lag_base)

    if verbose:
        print(f"Created lags in DataFrame: \n"
              f"    {lag_names} \n"
              f"with lag_base {lag_base}, min_lag {min_lag} (entries), max_lag {max_lag} (entries).")

    return dataframe_new, min_lag, lag_names


def data_split_features_labels(dataframe: pd.DataFrame, value_name: str, verbose=True) -> tuple:
    """
    Splits dataframe into two parts:
        features pd.DataFrame:      contains everything except the column value_name.
        labels pd.DataFrame:        contains only the column value_name.

    :param dataframe:   pd.DataFrame to split.
    :param value_name:  name of the column to isolate.
    :param verbose:     prints debugging information.
    :return:            tuple of pd.DataFrame (in the order (features, labels)).
    """
    if verbose:
        print(f"Splitting DataFrame into two parts by isolating the '{value_name}' column.")
    if value_name not in dataframe.columns:
        raise KeyError(f"Could not find the column '{value_name}' in the DataFrame.")
    X_data = dataframe.drop([value_name], axis=1).copy()
    y_data = dataframe[value_name].copy()
    return X_data, y_data


def data_split_train_test_valid(dataframe: pd.DataFrame, ratio: list[float], verbose=True):
    """
    Splits a pd.DataFrame into three parts (training, testing, validation) based on provided ratios.
    Ratios given must add to 1.

    :param dataframe:       pd.DataFrame to split.
    :param ratio:           a list of floats (adding to 1) for split ratios (train, test, valid).
    :param verbose:         prints progress to console.
    :return:                tuple of pd.DataFrame split in the order (train, test, valid).
    """
    if verbose:
        print(f"Splitting dataset into training, testing and validation with ratio {ratio}.")
    if int(math.fsum(ratio)) != 1:
        raise ValueError(f"Ratios must sum to 1. Current sum: {int(math.fsum(ratio))}.")

    # Calculating sizes:
    dataset_len = dataframe.shape[0]
    train_size = int(dataset_len * ratio[0])
    test_size = int(dataset_len * ratio[1])
    valid_size = int(dataset_len * ratio[2])

    # Splitting by sizes:
    data_train = dataframe.iloc[:train_size] if (train_size > 0) else None
    data_test = dataframe.iloc[train_size:train_size + test_size] if (test_size > 0) else None
    data_valid = dataframe.iloc[-valid_size:] if (valid_size > 0) else None

    return data_train, data_test, data_valid


def data_create_future(dataframe, datetime_name: str, value_name: str,
                       window_size: int, step="1h", append=True, verbose=True):
    """
    Consumes a pd.DataFrame, and creates a "future" DataFrame of (only) the DateTime and Target columns.
    All Target values in the value_name column are initialized to 0.

    If append=True, then the future DataFrame is appended to the original DataFrame.

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


def data_get_last_n(dataframe: pd.DataFrame, window_base=1, window_multiple=None, verbose=True):
    """
    Consumes a pd.DataFrame, a window_base and a window_multiple to get the last "n" (window_base * window_multiple)
    entries from the provided DataFrame. If the DataFrame is not large enough, the entire DataFrame is returned.

    If the DataFrame dataframe is None, then the function automatically returns None.

    :param dataframe:           DataFrame to get last entries from.
    :param window_base:         a base value for the window (set to 1 by default).
    :param window_multiple:     a window multiplier.
    :param verbose:             prints debugging information.
    :return:                    pd.DataFrame of last (window_base * window_multiple) entries of dataframe.
    """
    if verbose:
        print(f"Getting last {window_base * window_multiple} entries from the provided DataFrame.")
    if dataframe is None:
        print(f"SoftWarm: Trying to create last_n on a None DataFrame. Returning None.")
        return
    if window_base * window_multiple > dataframe.shape[0]:
        print(f"SortWarn: Trying to get {window_base * window_multiple} from a DataFrame of size "
              f"{dataframe.shape[0]}. Returning the entire DataFrame.")
        return dataframe
    if window_multiple is None:
        print("Window multiplier has not been set.")
        return
    dataframe_last_n = dataframe.iloc[-(window_base * window_multiple):].copy()
    return dataframe_last_n
