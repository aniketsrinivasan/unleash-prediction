# Information about Models
This directory stores model information for multiple machine learning architectures,
including XGBoost and LSTM. 

## Implementing Models

Each model is implemented as a class, typically in the formal "Model_vn" where Model
is the name of the underlying model architecture, and vn refers to the version number
(e.g. "XGBoostCV_v1" for XGBoost Cross-Validation Version 1).

Every model class must implement the following attributes and methods:
    
```
class Model_vn:
    __kwargs_hyperparams = dict(
        ...
    )
    
    def __init__(self, time_series: TimeSeries, ...):
        self.time_series = time_series
        self.target = ...       # [Target for prediction]
        self.features = ...     # [Features for prediction]
        
        self.model_name = "Model_vn"
        self.regressor = ...    # [Underlying model architecture]
        
    def train(self):
        ...
        
    def predict(self, custom_df=None, datetime_name=None, value_name=None):
        ...
        return predictions
```

These are required by the MasterModel and ModelTester frameworks to function.
Any additional implementation can be made as usual, provided these functions remain unaffected.