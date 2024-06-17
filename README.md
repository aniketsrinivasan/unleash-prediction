# Project Information

### Plan:
* Create a framework to manage time-series data efficiently and effectively
    for usage in model training, testing and validation.
  * Automate time-series data cleaning, augmentation (including feature creation) and splitting.
  * Experiment with various features (including DateTime feature extraction, time-series lags and rolling averages).
* Implement multiple models (XGBoost variants, LSTM variants, and combination models) comprehensively.
* Create an automated testing framework that compares model variants by accuracy and optimizes hyperparameters.
* Research into methods of combining deep learning frameworks with Fourier and Wavelet Transform to 
    encode frequency information for improved forecasting.

### Purpose:
* Research and implement a network (traffic) forecasting feature for Trisul (note: predictions can be made on any 
    continuous variable, not just traffic).
* Improved understanding of network traffic patterns.
* Provides additional data to better understand "expected" patterns to detect network anomalies.


### Updates:
Update (up to 17/06/2024):
* Implemented the following:
  * dataset_utils: a framework to manage (clean, create features, augment) raw data.
  * TimeSeries: a (wrapper) class that uses dataset_utils to work on .csv files.
  * Models (XGBoostCV, XGBoostTTV): custom-implemented (not pre-trained) XGBoost regressor models that train on 
      time-series data.
  * MasterModel: a (wrapper) class for model creation and management.
  * ModelTester: an automated testing framework to train, test and compare models on various metrics.


Update (17/06/2024):
* Restructured the implementation of Model(s) and MasterModel to remove inheritance (allows for ModelTester
    to work more efficiently).
* Implemented model_validation_loss() to calculate the loss (default "mean squared error" loss) of trained models
    on the validation set created in TimeSeries.
* Created ModelTester for test automation and comparison of various models.


# How to Use:
To be implemented.
