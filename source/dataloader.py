import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(
        self, dataframe, target, features, units, sequence_length=50, stride=1
    ):
        self.features = features
        self.target = target
        self.units = units
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()
        self.U = torch.tensor(dataframe[units].values).int()
        self.stride = stride

    def __len__(self):
        return self.X.shape[0] // self.stride

    def __getitem__(self, i):
        # Sequence is entirely within the data
        if i * self.stride >= self.sequence_length - 1:
            unit = self.U[
                i * self.stride - self.sequence_length + 1 : (i * self.stride + 1)
            ]
            cond = np.where(unit != unit[0])[0]

            if cond.size == 0:
                i_start = i * self.stride - self.sequence_length + 1
                x = self.X[i_start : (i * self.stride + 1), :]
            else:
                counter = cond[0]
                padding = self.X[
                    i * self.stride - (self.sequence_length - counter) + 1
                ].repeat(counter, 1)
                x = self.X[
                    i * self.stride - (self.sequence_length - counter) + 1 : (
                        i * self.stride + 1
                    ),
                    :,
                ]
                x = torch.cat((padding, x), 0)
        else:
            # Beginning of Sequence (backward filling)
            padding = self.X[0].repeat(self.sequence_length - i * self.stride - 1, 1)
            x = self.X[0 : (i * self.stride + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i * self.stride]


class SequenceDataset_with_Index(Dataset):
    def __init__(
        self,
        dataframe,
        target,
        features,
        units,
        sequence_length=5,
        stride=10,
        rul_scaler=92,
    ):
        self.features = features
        self.target = target
        self.units = units
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()
        self.U = torch.tensor(dataframe[units].values).int()
        self.stride = stride
        self.rul_scaler = rul_scaler

    def __len__(self):
        return self.X.shape[0] // self.stride

    def __getitem__(self, i):
        if i * self.stride >= self.sequence_length - 1:
            unit = self.U[
                i * self.stride - self.sequence_length + 1 : (i * self.stride + 1)
            ]
            cond = np.where(unit != unit[0])[0]

            if cond.size == 0:
                i_start = i * self.stride - self.sequence_length + 1
                x = self.X[i_start : (i * self.stride + 1), :]
            else:
                counter = cond[0]
                padding = self.X[
                    i * self.stride - (self.sequence_length - counter) + 1
                ].repeat(counter, 1)
                x = self.X[
                    i * self.stride - (self.sequence_length - counter) + 1 : (
                        i * self.stride + 1
                    ),
                    :,
                ]
                x = torch.cat((padding, x), 0)
        else:
            padding = self.X[0].repeat(self.sequence_length - i * self.stride - 1, 1)
            x = self.X[0 : (i * self.stride + 1), :]
            x = torch.cat((padding, x), 0)

        return x, torch.div(self.y[i * self.stride], self.rul_scaler), i


class SequenceDataset_oc(Dataset):
    def __init__(
        self,
        dataframe,
        target_rul,
        target_oc,
        features,
        units,
        sequence_length=50,
        stride=1,
    ):
        self.features = features
        self.target_rul = target_rul
        self.target_oc = target_oc
        self.units = units
        self.sequence_length = sequence_length
        self.y_rul = torch.tensor(dataframe[target_rul].values).float()
        self.y_oc = torch.tensor(dataframe[target_oc].values).float()
        self.X = torch.tensor(dataframe[features].values).float()
        self.U = torch.tensor(dataframe[units].values).int()
        self.stride = stride

    def __len__(self):
        return self.X.shape[0] // self.stride

    def __getitem__(self, i):
        if i * self.stride >= self.sequence_length - 1:
            unit = self.U[
                i * self.stride - self.sequence_length + 1 : (i * self.stride + 1)
            ]
            cond = np.where(unit != unit[0])[0]

            if cond.size == 0:
                i_start = i * self.stride - self.sequence_length + 1
                x = self.X[i_start : (i * self.stride + 1), :]
            else:
                counter = cond[0]
                padding = self.X[
                    i * self.stride - (self.sequence_length - counter) + 1
                ].repeat(counter, 1)
                x = self.X[
                    i * self.stride - (self.sequence_length - counter) + 1 : (
                        i * self.stride + 1
                    ),
                    :,
                ]
                x = torch.cat((padding, x), 0)
        else:
            padding = self.X[0].repeat(self.sequence_length - i * self.stride - 1, 1)
            x = self.X[0 : (i * self.stride + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y_rul[i * self.stride], self.y_oc[i * self.stride]


class SequenceDataset_oc_alt(Dataset):
    def __init__(
        self,
        dataframe,
        target_rul,
        target_oc,
        target_alt,
        features,
        units,
        sequence_length=50,
        stride=1,
    ):
        self.features = features
        self.target_rul = target_rul
        self.target_oc = target_oc
        self.target_alt = target_alt
        self.units = units
        self.sequence_length = sequence_length
        self.y_rul = torch.tensor(dataframe[target_rul].values).float()
        self.y_oc = torch.tensor(dataframe[target_oc].values).float()
        self.y_alt = torch.tensor(dataframe[target_alt].values).float()
        self.X = torch.tensor(dataframe[features].values).float()
        self.U = torch.tensor(dataframe[units].values).int()
        self.stride = stride

    def __len__(self):
        return self.X.shape[0] // self.stride

    def __getitem__(self, i):
        if i * self.stride >= self.sequence_length - 1:
            unit = self.U[
                i * self.stride - self.sequence_length + 1 : (i * self.stride + 1)
            ]
            cond = np.where(unit != unit[0])[0]

            if cond.size == 0:
                i_start = i * self.stride - self.sequence_length + 1
                x = self.X[i_start : (i * self.stride + 1), :]
            else:
                counter = cond[0]
                padding = self.X[
                    i * self.stride - (self.sequence_length - counter) + 1
                ].repeat(counter, 1)
                x = self.X[
                    i * self.stride - (self.sequence_length - counter) + 1 : (
                        i * self.stride + 1
                    ),
                    :,
                ]
                x = torch.cat((padding, x), 0)
        else:
            padding = self.X[0].repeat(self.sequence_length - i * self.stride - 1, 1)
            x = self.X[0 : (i * self.stride + 1), :]
            x = torch.cat((padding, x), 0)

        return (
            x,
            self.y_rul[i * self.stride],
            self.y_oc[i * self.stride],
            self.y_alt[i * self.stride],
        )
