from model_framework import *
from utils import TimeSeries
import matplotlib.pyplot as plt

# ~~~~~~~~~~~~~~~~~ HYPERPARAMETERS AND OTHER INFO ~~~~~~~~~~~~~~~~~
#                           FOR THE USER

# =========================  PARAMETERS  ===========================
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
# ==================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def main():
    # To-do:
    #   * add method to vary split_ratio based on future_window_size (so that valid is same size)
    #   * add XGBoostCV_v2 (another method of cross-validation) using sk-learn (?)
    #   * experiment with the following model_framework (hopefully on other datasets, too):
    #       - other XGBoost variants and hyperparameters
    #       - transformer? (this doesn't look promising though)
    #       - fourier analytic methods (i.e. wavelet transform?)
    #           the idea is to try encoding the frequencies into the training data directly
    #           most likely would require modifying feature creation (to include fourier)
    # idea: use wavelet transformations to initially train LSTM, and then fine-tune on dataset
    #           (almost ensures convergence to a meaningful pattern)
    # implement dataset methods to use torch.DataLoader for large datasets (for LSTM effiency)
    # rewrite significant code from scratch to restructure (add inheritance?)
    # implement saving and loading models properly
    # find ways to combine XGBoost and LSTM to come up with predictions.

    time_series = TimeSeries(**__kwargs_timeseries_init)
    time_series.prepare_from_scratch()
    print(time_series)
    print(time_series.value_name)

    model = MasterModel(time_series, "XGBoostTTV_v1",
                        read_stub=None,
                        write_stub=None,
                        is_trained=False)
    model.model_create()
    model.model_train()
    model.model_get_validation_loss()

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
