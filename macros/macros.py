# File for defining macros accessed by various aspects of model_framework.
# The alternative is to define macros using a C module, which is not optimized for
#   efficiency, and this may be faster.

# LOOKBACK for the LSTM model. Do NOT modify unless re-training the LSTM. This value must match
#   the sequential input size for the current LSTM model being used.
TorchLSTM_v2_LOOKBACK = 500     # length of sequential input to TorchLSTM_v2
