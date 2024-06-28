from ..dataset_utils import data_read_csv
from ..time_series import TimeSeries
import pandas as pd

# Initialization parameters for TimeSeries future preparation:
__kwargs_timeseries_future = dict(
    window_size=24*7,
    step_size="1h",
)
# Arguments to modify the creation of time-series features:
__kwargs_features = dict(
    minutes=False,
    hours=True,
    days_of_week=True,
    days_of_week_onehot=True,
    weeks=True,
    days_of_month=True,
    months=True,
    rolling_windows=[100],
    holidays_country=None,
    holidays_province=None,
)
# Arguments to modify the lag values created as features:
__lag_base = 24*7
__lag_multiples = [6, 12]
__lag_label = "m"
__kwargs_lags = dict(
    lag_base=__lag_base,
    lag_multiples=__lag_multiples,
    lag_label=__lag_label
)
# Last "n" kwargs:
__kwargs_last_n = dict(
    window_base=1,
    window_multiple=500+1,
)
__kwargs_timeseries_init = dict(
    csv_path=f"utils/tests/test_data.csv",
    datetime_name="Datetime",
    datetime_format="%Y-%m-%d %H:%M:%S",
    value_name="PJMW_MW",
    split_ratio=[0.7, 0.2, 0.1],
    kwargs_features=__kwargs_features,
    kwargs_lags=__kwargs_lags,
    kwargs_last_n=__kwargs_last_n,
    kwargs_prepare_future=__kwargs_timeseries_future,
    verbose=False,
)


def test_time_series_init() -> None:
    """
    Testing TimeSeries initialization.
    :return:
    """
    time_series = TimeSeries(**__kwargs_timeseries_init)
    assert isinstance(time_series, TimeSeries)
    assert time_series.df_augmented is None
    # Testing immediate lag parameter recognition:
    assert time_series.lag_min == 24*7*6
    assert time_series.lag_max == 24*7*12


def test_time_series_df_augment_none() -> None:
    """
    Testing default TimeSeries augmentation.
    :return:
    """
    time_series = TimeSeries(**__kwargs_timeseries_init)
    time_series.df_augment(update_self=False, override=False)
    assert time_series.df_augmented is None


def test_time_series_df_augment_custom() -> None:
    """
    Testing custom TimeSeries augmentation.
    :return:
    """
    test_df = data_read_csv("utils/tests/test_data.csv", verbose=False)
    time_series = TimeSeries(**__kwargs_timeseries_init)
    df, features, lags, lag_min = time_series.df_augment(custom_df=test_df.iloc[:3000], update_self=True,
                                                         override=False,
                                                         kwargs_features=__kwargs_features,
                                                         kwargs_lags=__kwargs_lags)
    assert time_series.df_augmented.shape[0] == 3000
    assert time_series.df_augmented_values_only is not None
    assert time_series.features == list(features.values())
    assert time_series.lags == lags
    assert time_series.lag_min == lag_min


def test_time_series_df_augment_override() -> None:
    """
    Testing the overriding feature for TimeSeries augmentation.
    :return:
    """
    time_series = TimeSeries(**__kwargs_timeseries_init)
    time_series.df_augmented = "Something's here!"
    # Creating augmented DataFrame, except df_augmented already exists:
    time_series.df_augment(update_self=True, override=False,
                           kwargs_features=__kwargs_features, kwargs_lags=__kwargs_lags)
    assert time_series.df_augmented == "Something's here!"
    time_series.df_augment(update_self=True, override=True,
                           kwargs_features=__kwargs_features, kwargs_lags=__kwargs_lags)
    assert isinstance(time_series.df_augmented, pd.DataFrame)


def test_time_series_prepare_for_forecast() -> None:
    """
    Testing the optimization feature for forecast preparation.
    :return:
    """
    time_series = TimeSeries(**__kwargs_timeseries_init)
    time_series.prepare_for_forecast()
    assert ((time_series.df_split_train is None) and
            (time_series.df_split_test is None) and
            (time_series.df_split_valid is None))
    assert isinstance(time_series.df_future_only, pd.DataFrame)
    # We are only collecting as many values as necessary:
    assert time_series.df_raw.shape[0] <= time_series.lag_max
