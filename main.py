from models import *
from utils import TimeSeries


# To modify, for the user:
__kwargs_features = dict()
__kwargs_lags = dict()


def main():
    # To-do:
    #   * add method to vary split_ratio based on future_window_size (so that valid is same size)
    #   * add XGBoostCV_v2 (another method of cross-validation) using sk-learn (?)
    #   * experiment with the following models (hopefully on other datasets, too):
    #       - other XGBoost variants and hyperparameters
    #       - LSTM (would need to build using PyTorch)
    #       - transformer? (this doesn't look promising though)
    #       - fourier analytic methods (i.e. wavelet transform?)
    #           the idea is to try encoding the frequencies into the training data directly
    #           most likely would require modifying feature creation (to include fourier)
    #   * implement rolling average feature in dataset_utils

    kwargs_timeseries_init = dict(csv_path="/Users/aniket/PycharmProjects/unleashPredictions/energy_data.csv",
                                  datetime_name="Datetime",
                                  datetime_format="%Y-%m-%d %H:%M:%S",
                                  value_name="PJMW_MW",
                                  verbose=True)

    kwargs_timeseries_prepare = dict(future_window_size=24*7*4,
                                     future_step_size="1h",
                                     split_ratio=[0.9, 0.05, 0.05],
                                     kwargs_features=None,
                                     kwargs_lags=None)

    tester = ModelTester(kwargs_timeseries_init, kwargs_timeseries_prepare)
    tester.create_model_dict(train=True, validate=True)

    # model = XGBoostTTV_v1(time_series=time_series)
    # model.train()
    # result = model.predict()
    # print(result[:10])
    # loss, loss_df = validation_loss(model)


if __name__ == '__main__':
    main()
