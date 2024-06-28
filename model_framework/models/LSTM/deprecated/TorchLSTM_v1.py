import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from utils import TimeSeries


class BaseLSTM_v1(nn.Module):
    # Hyperparameters for the LSTM:
    __INPUT_SIZE = 50
    __HIDDEN_SIZE = 512
    __NUM_LAYERS = 3
    __DROPOUT = 0.2
    # Size of the fully connected layer (at the end):
    __FULLY_CONNECTED_SIZE = 256
    # LSTM hyperparameters a dictionary:
    __kwargs_hyperparams = dict(
        input_size=__INPUT_SIZE,          # number of expected input (x) features (essentially sequence length
        hidden_size=__HIDDEN_SIZE,
        num_layers=__NUM_LAYERS,
        dropout=__DROPOUT,
    )
    __DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, num_classes):
        """
        Creation of the LSTM backbone that TorchLSTM_v1 functions on.

        :param num_classes:     number of outputs desired (per sequence input).
        """
        super(BaseLSTM_v1, self).__init__()
        # Setting instance information:
        self.num_classes = num_classes
        self.input_size = self.__INPUT_SIZE     # accessed by TorchLSTM_v1 later

        # Initializing the LSTM layer:
        self.lstm = nn.LSTM(batch_first=True, bidirectional=False, **self.__kwargs_hyperparams)
        # Initializing other layers (namely FC and ReLU layers) as a Sequential network:
        #   hidden_size ==> fully_connected_size ==> num_classes
        self.sequential = nn.Sequential(nn.ReLU(),
                                        nn.Linear(self.__HIDDEN_SIZE, self.__FULLY_CONNECTED_SIZE),
                                        nn.ReLU(),
                                        nn.Linear(self.__FULLY_CONNECTED_SIZE, num_classes)
                                        )

    # Implementation of forward-propagation method:
    def forward(self, x):
        """
        Forward method for BaseLSTM. The input "x" must be prepared using the data preparation function in
        TorchLSTM_v1.prepare_data() first.

        Here, "x" must be of shape (batch_size, sequence_length, input_size).

        :param x:       input for prediction, of the form returned by TorchLSTM_v1.prepare_data().
        :return:        predictions (in the form of a torch.float32
        """
        # Convert to type float32 (required by the LSTM model):
        x = x.to(torch.float32)

        # Initializing hidden state and internal state:
        #   hidden state:   (num_layers, x_size, hidden_size)
        h_0 = Variable(torch.zeros(self.__NUM_LAYERS, x.size(dim=0), self.__HIDDEN_SIZE))
        #   internal state: (num_layers, x_size, hidden_size)
        c_0 = Variable(torch.zeros(self.__NUM_LAYERS, x.size(dim=0), self.__HIDDEN_SIZE))

        # Propagating input through LSTM:
        x, (_, _) = self.lstm(x, (h_0.detach(), c_0.detach()))
        x = x.view(-1, self.__HIDDEN_SIZE)        # reshaping for Dense layer:  (, hidden_size)
        # Passing through Dense layers (as Sequential):
        output = self.sequential(x)
        return output


class TorchLSTM_v1:
    # LSTM hyperparameters:
    __NUM_CLASSES = 1            # how many output features are desired
    __LEARNING_RATE = 0.0001     # learning rate for optimizer (Adam)
    __EPOCHS = 1000
    __DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # How far to predict into the future (in entries), used by TorchLSTM_v1.predict():
    __N_FUTURE = 50

    def __init__(self, time_series: TimeSeries, read_from_stub=None, write_to_stub=None):
        # Storing this TimeSeries and its data information:
        self.time_series = time_series
        self.features = None        # this LSTM only uses the Target sequence, no other features
        self.target = time_series.value_name
        self.verbose = time_series.verbose

        self.model_name = "TorchLSTM_v1"
        self.read_from_stub = read_from_stub
        self.write_to_stub = write_to_stub
        # Initializing the regressor and other model information:
        self.regressor = BaseLSTM_v1(self.__NUM_CLASSES).to(self.__DEVICE)
        if (read_from_stub is not None) and os.path.exists(read_from_stub):
            print(f"Loading existing model from {read_from_stub}...")
            self.regressor.load_state_dict(torch.load(read_from_stub))

        self.loss_function = nn.MSELoss()       # mean-squared error for regression
        self.optimizer = torch.optim.Adam(self.regressor.parameters(), lr=self.__LEARNING_RATE)
        # How far to look back each prediction (stored as BaseLSTM.input_size):
        self.__lookback = self.regressor.input_size

        # Creating the Scaler used (for numerical stability with activation functions):
        self.scaler = StandardScaler()

    def prepare_data(self, dataset=None, value_name=None):
        """
        Prepares the data to work with the BaseLSTM architecture. By default, the dataset used is
        the training split (TimeSeries.df_split_train) of this TimeSeries.

        For custom datasets, the dataset's Target column name must be passed into value_name.
        Custom datasets are assumed to be sorted in time-series order.

        This function converts the dataset to sequential Variables to pass into the LSTM model.
        It returns two Variables:   (sequences, labels),    where sequences[i] has label labels[i].

        :param dataset:         passing a custom dataset.
        :param value_name:      name of the Target column (required if dataset is used).
        :return:                tuple[torch.Variable, torch.Variable]
        """
        # Initializing empty lists for inputs and labels:
        inputs, labels = [], []
        if dataset is None:
            dataset = self.time_series.df_split_train
            value_name = self.target
        elif value_name is None:
            raise IndexError(f"Custom dataset was provided, but without a Target column name.")
        # If the lookback is larger than the dataset itself, then end:
        if dataset.shape[0] <= self.__lookback:
            raise IndexError(f"Dataset ({dataset.shape[0]}) is <= the lookback ({self.__lookback}). "
                             f"Change input size (__INPUT_SIZE) in BaseLSTM implementation, or "
                             f"increase the number of entries provided for lookback.")
        # Conversion of the Target column into an array of shape (len(dataset), 1)
        try:
            dataset = np.array(dataset[value_name]).reshape(-1, 1)
        except Exception as e:
            raise IndexError(f"Error {e}. Does dataset contain a column {value_name}?")
        # Applying the Scaler transformation to the dataset:
        dataset = self.scaler.fit_transform(dataset)

        # The labels set is offset from inputs by __NUM_CLASSES. We iterate:
        for i in range(len(dataset) - self.__lookback - (self.__NUM_CLASSES - 1)):
            # Creating this feature (of length __lookback):
            feature = dataset[i : (i+self.__lookback)]
            # Creating this target (of length __lookback) incremented by __NUM_CLASSES:
            target = dataset[(i+self.__lookback) : (i+self.__lookback+self.__NUM_CLASSES)]
            # Appending:
            inputs.append(feature)
            labels.append(target)

        # Converting data into the desired format (as torch.Variable objects, reshaped):
        X_train, y_train = Variable(torch.tensor(np.array(inputs))), Variable(torch.tensor(np.array(labels)))
        # X_train:      ==> (number_of_batches, 1, batch_size)
        X_train = torch.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        # y_train       ==> (number_of_batches, num_classes)
        y_train = torch.reshape(y_train, (y_train.shape[0], y_train.shape[1]))
        # Conversion of X_train, y_train to torch.float32 (so they are both the same dtype):
        X_train, y_train = X_train.to(torch.float32), y_train.to(torch.float32)

        return X_train, y_train

    # Training method defaults to using the TimeSeries training split.
    def train(self):
        """
        Train the LSTM model in BaseLSTM. Uses the TimeSeries training data split.

        :return:    None.
        """
        # Prepare the training data (prepare_data uses time_series.df_split_train by default):
        X_train, y_train = self.prepare_data()
        print(X_train.shape)
        print(self.regressor)

        # Training loop:
        for epoch in range(self.__EPOCHS):
            output = self.regressor.forward(X_train)    # forward-propagation
            self.optimizer.zero_grad()                  # zero-grad

            # Calculating loss and performing back-propagation:
            loss = self.loss_function(output, y_train)
            loss.backward()
            self.optimizer.step()

            # Printing training loop information (every 100 epochs):
            if (self.verbose) and (epoch % 100 == 0):
                print(f"Epoch: {epoch}, Loss: {loss.item():.5f}")

        # Saving model:
        if (self.write_to_stub is not None) and os.path.exists(self.write_to_stub):
            torch.save(self.regressor.state_dict(), self.write_to_stub)
            print(f"Saving trained model to {self.write_to_stub}.")

        return

    def predict(self, custom_df=None, value_name=None, datetime_name=None):
        """
        Running predictions on a dataset. Uses the validation data split (df_split_valid) by default.
        To use a custom dataset, pass the pd.DataFrame and Target column name.

        :param custom_df:       custom dataset (pd.DataFrame).
        :param value_name:      name of Target column for dataset.
        :param datetime_name:   name of DateTime column for dataset.
        :return:                predictions (as a numpy array) that have been re-scaled.
        """
        if custom_df is None:
            custom_df = self.time_series.df_split_valid_last_n
            value_name = self.target
        dataset, _ = self.prepare_data(dataset=custom_df, value_name=value_name)
        # Getting predictions, detaching to numpy (to allow pass through inverse_transform()):
        with torch.no_grad():
            predictions = self.regressor.forward(dataset).detach().numpy()
        print(predictions)
        predictions = self.scaler.inverse_transform(predictions)
        print(predictions)
        return predictions
