from model_framework import *
from utils import TimeSeries


# To modify, for the user:
__kwargs_features = dict(
    hours=True,
    days_of_week=True,
    weeks=True,
    days_of_month=False,
    months=False
)

__lag_base = 24
__lag_multiples = [1, 7, 7*4, 7*12, 7*54]
__lag_label = "d"
__kwargs_lags = dict(
    lag_base=__lag_base,
    lag_multiples=__lag_multiples,
    lag_label=__lag_label
)

__kwargs_timeseries_init = dict(
    csv_path="/Users/aniket/PycharmProjects/unleashPredictions/energy_data.csv",
    datetime_name="Datetime",
    datetime_format="%Y-%m-%d %H:%M:%S",
    value_name="PJMW_MW",
    verbose=True
)

__kwargs_timeseries_prepare = dict(
    future_window_size=24*7*4,
    future_step_size="1h",
    split_ratio=[0.9, 0.05, 0.05],
)


def main():
    # To-do:
    #   * add method to vary split_ratio based on future_window_size (so that valid is same size)
    #   * add XGBoostCV_v2 (another method of cross-validation) using sk-learn (?)
    #   * experiment with the following model_framework (hopefully on other datasets, too):
    #       - other XGBoost variants and hyperparameters
    #       - LSTM (would need to build using PyTorch)
    #       - transformer? (this doesn't look promising though)
    #       - fourier analytic methods (i.e. wavelet transform?)
    #           the idea is to try encoding the frequencies into the training data directly
    #           most likely would require modifying feature creation (to include fourier)
    #   * implement rolling average feature in dataset_utils

    tester = ModelTester(__kwargs_timeseries_init, __kwargs_timeseries_prepare,
                         __kwargs_features, __kwargs_lags)
    tester.create_model_dict(train=False, validate=False)
    tester.run_training()
    tester.run_validation(verbose=True)
    tester.get_validation_losses()

    # model = XGBoostTTV_v1(time_series=time_series)
    # model.train()
    # result = model.predict()
    # print(result[:10])
    # loss, loss_df = validation_loss(model)


if __name__ == '__main__':
    main()
