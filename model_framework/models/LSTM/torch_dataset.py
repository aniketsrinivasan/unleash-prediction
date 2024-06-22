import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader


# Implementing a Torch Dataset for our LSTM model.
class TorchDataset(Dataset):
    def __init__(self, inputs, labels):
        """
        Inputs and labels are fed into TorchDataset as arrays such that inputs[i] and labels[i]
        correspond to one another.

        :param inputs:  array of inputs.
        :param labels:  array of labels.
        """
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __get__(self, idx):
        sequence, label = self.inputs[idx], self.labels[idx]
        return dict(
            sequence=torch.Tensor(sequence.to_numpy()),
            label=torch.Tensor(label).float()
        )


class TorchDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size=16):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size

        self.train_dataset = None
        self.test_dataset = None

    def setup(self):
        self.train_dataset = TorchDataset(self.train_sequences[0], self.train_sequences[1])
        self.test_dataset = TorchDataset(self.test_sequences[0], self.test_sequences[1])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )


