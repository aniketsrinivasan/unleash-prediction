import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from utils import TimeSeries


class BaseLSTM(nn.Module):
    # Hyperparameters for the LSTM:
    __kwargs_hyperparams = dict(
        input_size=50,       # number of expected input (x) features
        hidden_size=128,     # number of features in the hidden state
        num_layers=2,       # number of RNNs to stack (forming a stacked LSTM)
        dropout=0.2,
    )
    # Extracting only the individual parameters:
    __INPUT_SIZE = __kwargs_hyperparams["input_size"]
    __HIDDEN_SIZE = __kwargs_hyperparams["hidden_size"]
    __NUM_LAYERS = __kwargs_hyperparams["num_layers"]
    # Size of the fully connected layer:
    __FULLY_CONNECTED_SIZE = 128

    def __init__(self, num_classes, seq_length):
        """
        Creation of the LSTM backbone that TorchLSTM_v1 functions on.

        :param num_classes:     number of classes.
        :param seq_length:      length of sequence
        """
        super(BaseLSTM, self).__init__()
        # Setting instance information:
        self.num_classes = num_classes      # number of classes
        self.seq_length = seq_length        # size of the sequence

        # Storing input size (accessed by the main model later):
        self.input_size = self.__INPUT_SIZE

        # Initializing the LSTM model:
        self.lstm = nn.LSTM(batch_first=True, **self.__kwargs_hyperparams)
        # Initializing other layers (namely FC and ReLU layers):
        self.fc1 = nn.Linear(self.__HIDDEN_SIZE, self.__FULLY_CONNECTED_SIZE)
        self.fc2 = nn.Linear(self.__FULLY_CONNECTED_SIZE, num_classes)
        self.relu = nn.ReLU()

    # Forward method implementation:
    def forward(self, x):
        """
        Forward method for BaseLSTM. The input "x" is expected to be prepared using
        TorchLSTM_v1.prepare_data() first.

        :param x:       input for prediction.
        :return:        predictions (check dtype).
        """
        # Changing to type float32 (required by LSTM):
        x = x.to(torch.float32)
        # Initializing:
        #   hidden state:
        h_0 = Variable(torch.zeros(self.__NUM_LAYERS, x.size(0), self.__HIDDEN_SIZE))
        #   internal state:
        c_0 = Variable(torch.zeros(self.__NUM_LAYERS, x.size(0), self.__HIDDEN_SIZE))

        # Propagate input through LSTM:
        _, (hn, cn) = self.lstm(x, (h_0, c_0))
        # Reshaping data for the Dense layer:
        hn = hn.view(-1, self.__HIDDEN_SIZE)
        # Passing through Dense layers:
        out = self.relu(hn)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class TorchLSTM_v1:
    # LSTM hyperparameters:
    __NUM_CLASSES = 1
    __SEQ_LENGTH = 20
    # Learning rate and epochs for training:
    __LEARNING_RATE = 0.00001
    __EPOCHS = 5000

    def __init__(self, time_series: TimeSeries):
        # Storing this TimeSeries:
        self.time_series = time_series
        # Creating a list of features (as column names) for the dataset:
        #   note: LSTM only uses the Target sequence, so there are no features.
        self.features = None
        # Target to predict (as a column name):
        self.target = time_series.value_name

        self.model_name = "TorchLSTM_v1"
        # Initializing the model:
        self.regressor = BaseLSTM(self.__NUM_CLASSES, self.__SEQ_LENGTH)
        self.loss_function = nn.MSELoss()       # mean-squared error for regression
        self.optimizer = torch.optim.Adam(self.regressor.parameters(), self.__LEARNING_RATE)
        # How far to look back each prediction:
        self.__LOOKBACK = self.regressor.input_size

        # Creating the Scaler used (for numerical stability when using activation functions):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, dataset=None, dataset_value_name=None):
        """
        Creates two torch.Tensor objects containing inputs and labels of length self.__LOOKBACK from the
        TimeSeries's training data split.

        The dataset passed must be a pd.DataFrame.

        :param dataset:             preparing a custom dataset. if None, then self.time_series.df_split_train is used.
        :param dataset_value_name:  name of the Target column in dataset.
        :return:                    tuple[torch.tensor, torch.tensor].
        """
        # Initializing empty lists for inputs and labels (X and y):
        inputs, labels = [], []
        if dataset is None:
            dataset = self.time_series.df_split_train
            dataset = np.array(dataset[self.time_series.value_name]).reshape(-1, 1)
        else:
            dataset = np.array(dataset[dataset_value_name]).reshape(-1, 1)
        # If the lookback is larger than the dataset itself:
        if dataset.shape[0] < self.__LOOKBACK:
            raise IndexError(f"Dataset ({dataset.shape[0]}) is smaller than the lookback ({self.__LOOKBACK}).")

        dataset = self.scaler.fit_transform(dataset)
        for i in range(len(dataset) - self.__LOOKBACK):
            # Creating this feature (of length __LOOKBACK):
            feature = dataset[i : i+self.__LOOKBACK]
            # Creating this target (of length __LOOKBACK, but incremented by 1):
            target = dataset[i+1 : i+self.__LOOKBACK+1]
            # Appending:
            inputs.append(feature)
            labels.append(target)

        # Converting our data into the desired format (as torch.Variable objects, reshaped):
        X_train, y_train = Variable(torch.tensor(np.array(inputs))), Variable(torch.tensor(np.array(labels)))
        X_train = torch.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        y_train = torch.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
        return X_train, y_train

    def train(self):
        # Preparing the training data:
        X_train, y_train = self.prepare_data()
        #   outputs is of dtype="torch.float32" but y_train is of dtype="torch.float64"
        #   so we convert y_train to match (loss.backward() doesn't work otherwise):
        y_train = y_train.to(torch.float32)
        print(X_train.dtype)
        print(X_train.size())

        # Training loop:
        for epoch in range(self.__EPOCHS):
            outputs = self.regressor.forward(X_train)      # forward-propagation
            self.optimizer.zero_grad()

            # Calculating loss and performing back-propagation:
            loss = self.loss_function(outputs, y_train)
            loss.backward()
            self.optimizer.step()

            if (self.time_series.verbose) and (epoch % 100 == 0):
                print(f"Epoch: {epoch}, Loss: {loss.item():.5f}.")

    def predict(self, dataset=None, dataset_value_name=None):
        if dataset is None:
            dataset = self.time_series.df_split_train
            dataset_value_name = self.time_series.value_name
        dataset, _ = self.prepare_data(dataset, dataset_value_name)
        predictions = self.regressor.forward(dataset).detach().numpy()
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
