import matplotlib.pyplot as plt
import utils
from model_framework import HybridAveraging_v1
from model_framework import ModelTester
from main import __kwargs_timeseries_init

ROOT_DIR = "/Users/aniket/PycharmProjects/unleashPredictions/"

__models_init_dict = {
    "XGBoostTTV_v1": dict(
        read_from_stub=f"{ROOT_DIR}model_framework/models/xgboost/saved_models/xgboost_ttv_1",
        write_to_stub=f"{ROOT_DIR}model_framework/models/xgboost/saved_models/xgboost_ttv_1",
        is_trained=True,
    ),
    # "XGBoostCV_v1": dict(
    #     read_from_stub=None,
    #     write_to_stub=None,
    #     is_trained=False,
    # ),
    "TorchLSTM_v2": dict(
        read_from_stub=f"{ROOT_DIR}model_framework/models/LSTM/saved_models/lstm_energy_data_500",
        write_to_stub=None,
        is_trained=True,
    )
}

__kwargs_averaging = dict(
    isolate="Prediction_TorchLSTM_v2",
    ceil_floor_values=(0.7, 0.0),
    smooth_ratio=0.5,
)


def main():
    tester = ModelTester(kwargs_timeseries_init=__kwargs_timeseries_init,
                                         models_init=__models_init_dict)
    tester.create_model_dict()
    tester.run_training()
    tester.run_validation()
    # tester.get_validation_losses()
    # tester.plot_validation_losses()
    # tester.plot_validation_mean()
    tester.plot_validation_scheduler(isolate_model="TorchLSTM_v2")
    time_series = utils.TimeSeries(**__kwargs_timeseries_init)
    time_series.prepare_from_scratch()

    # Hybrid Model:
    hybrid = HybridAveraging_v1(time_series, __models_init_dict)
    predictions, column_name = hybrid.predict(averaging="average_linear_split", kwargs_averaging=__kwargs_averaging)

    avg_preds = predictions["average_linear_split"].copy()
    avg_preds.plot()
    plt.ylim(bottom=0)
    plt.show()


if __name__ == "__main__":
    main()
