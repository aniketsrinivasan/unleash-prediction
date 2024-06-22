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

Update (19/06/2024):
* Implemented a (very rough) TorchLSTM_v1 model (uses RNNs for prediction).
* Modified methods in TimeSeries and functions in dataset_utils to work with LSTMs.
* Ran initial training and testing with TorchLSTM_v1 (needs significant improvement).

Update (20/06/2024):
* Completed an initial TorchLSTM_v1 model which works well (trained and debugged).
* Solved LSTM problem where the model converged to the TimeSeries mean during training.
* Modified methods in TimeSeries to work with LSTMs.
* Began implementation of TorchLSTM_v1 into ModelTester testing framework.

Update (21/06/2024):
* Overhauled implementation of dataset_utils and TimeSeries to streamline the full-scale
    implementation of TorchLSTM_v1.
* Updated main.py hyperparameters to fit new TimeSeries implementation.

Update (22/06/2024):
* Updated model testing framework to plot predictions and accomodate model initialization
    parameters on a MasterModel-basis.
* Implemented TorchLSTM_v2:
  * TorchLSTM_v1: All-in-one data training. Predictions are not recursive (multi-step).
  * TorchLSTM_v2: Batch-wise training with custom batch sizes (suited for large datasets). 
      Predictions are recursive (multi-step).


# How to Use:
To be implemented.
