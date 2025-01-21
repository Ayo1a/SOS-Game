# Dataset for pre-training
import torch
from torch.utils.data import Dataset

class GameDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, policy, value = self.data[idx]
        return torch.tensor(state, dtype=torch.float32), \
               torch.tensor(policy, dtype=torch.float32), \
               torch.tensor(value, dtype=torch.float32)