import torch
from torch.utils.data import Dataset

class MalariaDataset(Dataset):
    def __init__(self, dataframe, input_size):
        self.X = torch.tensor(dataframe.iloc[:, :input_size].values, dtype=torch.float32)
        self.Y = torch.tensor(dataframe.iloc[:, input_size:].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]