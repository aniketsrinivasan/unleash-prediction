import matplotlib.pyplot as plt
from model_framework import *
from utils import TimeSeries
from macros import TorchLSTM_v2_LOOKBACK

# ~~~~~~~~~~~~~~~~~ HYPERPARAMETERS AND OTHER INFO ~~~~~~~~~~~~~~~~~
#                           FOR THE USER
# =========================  PARAMETERS  ===========================
# Initialization parameters for TimeSeries future preparation:
__kwargs_timeseries_future = dict(
    window_size=60*24,
    step_size="1min",
)

# =======================  HYPERPARAMETERS  ========================
# Arguments to modify the creation of time-series features:
__kwargs_features = dict(
    minutes=True,
    hours=True,
    days_of_week=True,
    days_of_week_onehot=True,
    weeks=True,
    days_of_month=True,
    months=True,
    rolling_windows=[100],
    holidays_country="IND",
    holidays_province=None,
)
# Arguments to modify the lag values created as features:
__lag_base = 60*24*7
__lag_multiples = [1, 2, 3, 4, 5]
__lag_label = "w"
__kwargs_lags = dict(
    lag_base=__lag_base,
    lag_multiples=__lag_multiples,
    lag_label=__lag_label
)

# Last "n" kwargs:
__kwargs_last_n = dict(
    window_base=1,
    window_multiple=TorchLSTM_v2_LOOKBACK+1,
)

__kwargs_timeseries_init = dict(
    csv_path=f"dataset.csv",
    datetime_name="Datetime",
    datetime_format="Unix",
    value_name="Target",
    split_ratio=[0.9, 0.01, 0.09],
    kwargs_features=__kwargs_features,
    kwargs_lags=__kwargs_lags,
    kwargs_last_n=__kwargs_last_n,
    kwargs_prepare_future=__kwargs_timeseries_future,
    verbose=True,
)
# ==================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def main():
    # To-do:
    #   * add XGBoostCV_v2 (another method of cross-validation) using sk-learn (?)
    #   * experiment with the following model_framework (hopefully on other datasets, too):
    #       - other XGBoost variants and hyperparameters
    #       - fourier analytic methods (i.e. wavelet transform?)
    #           the idea is to try encoding the frequencies into the training data directly
    #           most likely would require modifying feature creation (to include fourier)
    # idea: use wavelet transformations to initially train LSTM, and then fine-tune on dataset
    #           (almost ensures convergence to a meaningful pattern)
    # implement saving and loading models properly
    # find ways to combine XGBoost and LSTM to come up with predictions.
    # implement multi-step predictions for XGBoost (so that lag sizes can be independent
    #   of the prediction window).

    # when taking average of all three models' predictions, try some "scheduling"-type average where:
    #   LSTM's importance takes precedence in the initial portion of predictions
    #   XGBoost's importance increases over length of predictions
    # find way to train XGBoost on ALL training data, and then predict on validation,
    #   but LSTM trains on a set quantity of "most recent" data (say 100k entries) and then
    #   predicts.
    # minute-wise predictions can be made by XGBoost, but hour-wise predictions by LSTM
    #   (this way 4 weeks of hourly predictions = 672 entries, which is a reasonable lookback).
    #   note: LSTM can be used to predict other low-frequency intervals similarly (e.g. day-wise)
    #         so the predictive power can really extend as far as necessary
    #   note: each interval would need its own trained LSTM model
    # hour-wise predictions can be fed into XGBoost as a feature, in order to predict minute-wise.

    time_series = TimeSeries(**__kwargs_timeseries_init)
    time_series.prepare_for_validation(is_sorted=True)
    print(time_series)
    print(time_series.value_name)

    model = MasterModel(time_series, "XGBoostTTV_v1",
                        read_from_stub="model_framework/models/xgboost/saved_models/xgboost_ttv_financial_1.ubj",
                        write_to_stub="model_framework/models/xgboost/saved_models/xgboost_ttv_financial_1.ubj",
                        is_trained=True)
    model.model_create()
    model.model_train()
    model.model_run_validation()

    # tester = ModelTester(__kwargs_timeseries_init, __kwargs_timeseries_prepare,
    #                      __kwargs_features, __kwargs_lags)
    # print(tester.time_series)
    # print(tester.time_series.df_augmented)
    # tester.create_model_dict(train=False, validate=False)
    # tester.run_training(verbose=False)
    # tester.run_validation(verbose=True)
    # tester.get_validation_losses()

    # model = XGBoostTTV_v1(time_series=time_series)
    # model.train()
    # result = model.predict()
    # print(result[:10])
    # loss, loss_df = validation_loss(model)


if __name__ == '__main__':
    main()
