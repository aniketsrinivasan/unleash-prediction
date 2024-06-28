from utils import (data_read_csv, data_datetime_conversion, data_create_features,
                   data_create_lags, data_datetime_sort, data_split_train_test_valid,
                   data_create_future, data_get_last_n)
import pytest
import pandas as pd


def test_data_read_csv_invalid() -> None:
    """
    Tests reading an invalid file name.
    :return:
    """
    with pytest.raises(FileNotFoundError):
        data_read_csv("invalid_file", verbose=False)
    with pytest.raises(FileNotFoundError):
        data_read_csv("$(#DJ this path cannot exist!", verbose=False)


def test_data_read_csv_valid() -> None:
    """
    Tests reading a valid file name into a pd.DataFrame.
    :return:
    """
    test_df = data_read_csv("utils/tests/test_data.csv", verbose=False)
    assert isinstance(test_df, pd.DataFrame)
    assert test_df.shape[1] == 2


def test_data_create_features_empty() -> None:
    """
    Tests creating no features (default).
    :return:
    """
    test_df = data_read_csv("utils/tests/test_data.csv", verbose=False)
    test_df, column_dict = data_create_features(test_df, datetime_name="Datetime", value_name="PJMW_MW",
                                                verbose=False)
    assert column_dict == {}
    assert test_df.shape[1] == 2


def test_data_datetime_conversion_valid() -> None:
    """
    Tests converting DateTime to pd.Timestamp.
    :return:
    """
    test_df = data_read_csv("utils/tests/test_data.csv", verbose=False)
    test_df = data_datetime_conversion(test_df, datetime_name="Datetime", datetime_format="%Y-%m-%d %H:%M:%S",
                                       verbose=False)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(test_df["Datetime"].iloc[0], pd.Timestamp)


def test_data_create_features_valid() -> None:
    """
    Tests creating multiple features.
    :return:
    """
    test_df = data_read_csv("utils/tests/test_data.csv", verbose=False)
    test_df = data_datetime_conversion(test_df, datetime_name="Datetime", datetime_format="%Y-%m-%d %H:%M:%S")
    test_df, column_dict = data_create_features(test_df, datetime_name="Datetime", value_name="PJMW_MW",
                                                minutes=False,
                                                hours=True,
                                                days_of_week=True,
                                                days_of_week_onehot=True,
                                                weeks=True,
                                                days_of_month=True,
                                                months=True,
                                                rolling_windows=[100],
                                                holidays_country="IND",
                                                holidays_province=None,
                                                verbose=False)
    assert len(list(column_dict.values())) == 14
    assert test_df.shape[1] == 16
    # Checks that 2002/12/31 was a Tuesday, and 24 hours before was a Monday:
    assert test_df[column_dict["tuesday"]].iloc[0] == 1
    assert test_df[column_dict["monday"]].iloc[24] == 1


def test_data_datetime_sort_invalid() -> None:
    test_df = data_read_csv("utils/tests/test_data.csv", verbose=False)
    test_df = data_datetime_conversion(test_df, datetime_name="Datetime", datetime_format="%Y-%m-%d %H:%M:%S",
                                       verbose=False)
    with pytest.raises(KeyError):
        data_datetime_sort(test_df, datetime_name="Invalid!", verbose=False)


def test_data_datetime_sort_valid() -> None:
    test_df = data_read_csv("utils/tests/test_data.csv", verbose=False)
    test_df = data_datetime_conversion(test_df, datetime_name="Datetime", datetime_format="%Y-%m-%d %H:%M:%S",
                                       verbose=False)
    test_df = data_datetime_sort(test_df, datetime_name="Datetime", verbose=False)
    assert test_df["Datetime"].iloc[0] < test_df["Datetime"].iloc[1]
    assert test_df["Datetime"].iloc[300] < test_df["Datetime"].iloc[301]
    assert test_df["Datetime"].iloc[1000] < test_df["Datetime"].iloc[1001]


def test_data_create_lags_invalid() -> None:
    test_df = data_read_csv("utils/tests/test_data.csv", verbose=False)
    test_df = data_datetime_conversion(test_df, datetime_name="Datetime", datetime_format="%Y-%m-%d %H:%M:%S",
                                       verbose=False)
    test_df = data_datetime_sort(test_df, datetime_name="Datetime", verbose=False)
    with pytest.raises(IndexError):
        test_df = data_create_lags(test_df, "PJMW_MW", lag_base=100, lag_multiples=[10, 100],
                                   verbose=False)


def test_data_create_lags_valid() -> None:
    test_df = data_read_csv("utils/tests/test_data.csv", verbose=False)
    test_df = data_datetime_conversion(test_df, datetime_name="Datetime", datetime_format="%Y-%m-%d %H:%M:%S",
                                       verbose=False)
    test_df = data_datetime_sort(test_df, datetime_name="Datetime", verbose=False)
    test_df, min_lag, lag_list = data_create_lags(test_df, "PJMW_MW",
                                                  lag_base=1, lag_multiples=[10, 100],
                                                  verbose=False)
    assert min_lag == 10
    assert len(lag_list) == 2
    # Checking that lagged columns exist and actually work:
    assert test_df["lag_10"].iloc[10] == test_df["PJMW_MW"].iloc[0]
    assert test_df["lag_10"].iloc[40] == test_df["PJMW_MW"].iloc[30]
    assert test_df["lag_100"].iloc[400] == test_df["PJMW_MW"].iloc[300]


def test_data_split_train_test_valid_invalid() -> None:
    test_df = data_read_csv("utils/tests/test_data.csv", verbose=False)
    with pytest.raises(ValueError):
        test_df = data_split_train_test_valid(test_df, [1, 0, 1], verbose=False)


def test_data_split_train_test_valid_valid() -> None:
    test_df = data_read_csv("utils/tests/test_data.csv", verbose=False)
    train, test, valid = data_split_train_test_valid(test_df, [0.5, 0.3, 0.2], verbose=False)
    assert train.shape[0] == test_df.shape[0] * 0.5
    assert test.shape[0] == test_df.shape[0] * 0.3
    assert valid.shape[0] == test_df.shape[0] * 0.2
    assert train.shape[0] + test.shape[0] + valid.shape[0] == test_df.shape[0]


def test_data_create_future_append_valid() -> None:
    test_df = data_read_csv("utils/tests/test_data.csv", verbose=False)
    future_appended = data_create_future(test_df, "Datetime", "PJMW_MW",
                                         500, step="1h", append=True, verbose=False)
    assert future_appended.shape[0] == test_df.shape[0] + 500
    assert future_appended.shape[1] == test_df.shape[1]


def test_data_create_future_valid() -> None:
    test_df = data_read_csv("utils/tests/test_data.csv", verbose=False)
    test_df = data_datetime_conversion(test_df, datetime_name="Datetime", datetime_format="%Y-%m-%d %H:%M:%S",
                                       verbose=False)
    test_df = data_datetime_sort(test_df, datetime_name="Datetime", verbose=False)
    future = data_create_future(test_df, "Datetime", "PJMW_MW",
                                         500, step="1h", append=False, verbose=False)
    assert future.shape[0] == 500
    # Checking that the first DateTime value of future is beyond the last value of test_df:
    assert future["Datetime"].iloc[0] > test_df["Datetime"].iloc[-1]


def test_data_get_last_n_valid() -> None:
    test_df = data_read_csv("utils/tests/test_data.csv", verbose=False)
    last_n = data_get_last_n(test_df, 1, 1000, verbose=False)
    assert last_n.shape[0] == 1000
    assert test_df["Datetime"].iloc[-1] == last_n["Datetime"].iloc[-1]
