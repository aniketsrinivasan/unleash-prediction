from utils import (data_read_csv, data_datetime_conversion, data_datetime_sort,
                   data_datetime_create_features, data_create_lags, data_split_features_labels,
                   data_split_train_test_valid, data_create_future)
from models import MasterModel
from models import (XGBoostCV_v1, XGBoostTTV_v1)
from utils import TimeSeries


def main():
    time_series = TimeSeries(csv_path="/Users/aniket/PycharmProjects/unleashPredictions/energy_data.csv",
                             datetime_name="Datetime",
                             datetime_format="%Y-%m-%d %H:%M:%S",
                             value_name="PJMW_MW",
                             verbose=True)
    time_series.prepare_from_scratch(future_window_size=24*7*4,
                                     future_step_size="1h",
                                     split_ratio=[0.7, 0.2, 0.1],
                                     kwargs_features=None,
                                     kwargs_lags=None)
    print(time_series)

    # model = XGBoostCV_v1(time_series=time_series)
    # model.train()
    # result = model.predict()
    # print(result)

    model = XGBoostTTV_v1(time_series=time_series)
    model.train()
    result = model.predict()
    print(result)


if __name__ == '__main__':
    main()
