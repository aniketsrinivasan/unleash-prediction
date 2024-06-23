import model_framework
from main import __kwargs_timeseries_init

ROOT_DIR = "/Users/aniket/PycharmProjects/unleashPredictions/"

__models_init_dict = {
    "XGBoostTTV_v1": dict(
        read_from_stub=None,
        write_to_stub=None,
        is_trained=False,
    ),
    "XGBoostCV_v1": dict(
        read_from_stub=None,
        write_to_stub=None,
        is_trained=False,
    ),
    "TorchLSTM_v2": dict(
        read_from_stub=f"{ROOT_DIR}model_framework/models/LSTM/saved_models/lstm_energy_data_500",
        write_to_stub=None,
        is_trained=True,
    )
}


def main():
    tester = model_framework.ModelTester(kwargs_timeseries_init=__kwargs_timeseries_init,
                                         models_init=__models_init_dict)
    tester.create_model_dict()
    tester.run_training()
    tester.run_validation()
    tester.get_validation_losses()
    tester.plot_validation_losses()
    tester.plot_validation_mean()
    tester.plot_validation_scheduler(isolate_model="TorchLSTM_v2")


if __name__ == "__main__":
    main()
