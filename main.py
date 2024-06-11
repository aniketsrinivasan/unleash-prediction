from utils import (data_read_csv, data_datetime_conversion, data_datetime_sort,
                            data_datetime_create_features, data_create_lags, data_split_features_labels,
                            data_split_train_test_valid, data_create_future)
from models import ModelXGBoostCV


def main():
    dataset = data_read_csv("/Users/aniket/PycharmProjects/unleashPredictions/energy_data.csv")
    dataset = data_datetime_conversion(dataset, "Datetime", "%Y-%m-%d %H:%M:%S")
    dataset = data_datetime_sort(dataset, "Datetime")
    dataset, features = data_datetime_create_features(dataset, "Datetime")
    dataset, _ = data_create_lags(dataset,
                                  value_name="PJMW_MW",
                                  lag_base=24*7,
                                  lag_multiples=[4, 8, 12, 28, 52],
                                  lag_label="w")
    data_train, _, data_valid = data_split_train_test_valid(dataset,
                                                            ratio=[0.8, 0, 0.2])

    model_cv = ModelXGBoostCV(dataframe=data_train,
                              features=list(features.values()),
                              target="PJMW_MW")

    model_cv.train_cv(test_size=None)


if __name__ == '__main__':
    main()
