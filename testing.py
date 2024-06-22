from model_framework import MasterModel2


# Initialization parameters for TimeSeries future preparation:
__kwargs_timeseries_future = dict(
    window_size=24*7*4,
    step_size="1h",
)

# =======================  HYPERPARAMETERS  ========================
# Arguments to modify the creation of time-series features:
__kwargs_features = dict(
    hours=False,
    days_of_week=True,
    weeks=True,
    days_of_month=True,
    months=True,
    rolling_windows=[100],
    holidays_country=None,
    holidays_province=None,
)
# Arguments to modify the lag values created as features:
__lag_base = 28
__lag_multiples = [3, 6, 12]
__lag_label = "d"
__kwargs_lags = dict(
    lag_base=__lag_base,
    lag_multiples=__lag_multiples,
    lag_label=__lag_label
)

# Last "n" kwargs:
__kwargs_last_n = dict(
    window_base=1,
    window_multiple=1
)

__kwargs_timeseries_init = dict(
    csv_path="/Users/aniket/PycharmProjects/unleashPredictions/Electric_Production.csv",
    datetime_name="DATE",
    datetime_format="%d/%m/%Y",
    value_name="IPG2211A2N",
    split_ratio=[0.7, 0.1, 0.2],
    kwargs_features=__kwargs_features,
    kwargs_lags=__kwargs_lags,
    kwargs_last_n=__kwargs_last_n,
    kwargs_prepare_future=__kwargs_timeseries_future,
    verbose=True,
)


def main():
    model = MasterModel2(model_name="meow", kwargs_timeseries_init=__kwargs_timeseries_init)
    print(model)


if __name__ == "__main__":
    main()