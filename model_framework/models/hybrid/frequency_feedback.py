from ..LSTM import TorchLSTM_v2
from ..xgboost import XGBoostTTV_v1


# Using a novel frequency-based feedback system that combines XGBoostTTV_v1 and TorchLSTM_v2
#   to predict on datasets with low correlation to wave-like behaviours.
# The dataset is split into low-frequency and high-frequency portions, which are then predicted
#   on in a feedback system between LSTM (low-frequency) and XGBoost (high-frequency).


class HybridAveraging_v1:
    def __init__(self):
        pass