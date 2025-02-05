# Dataset for pre-training
'''the goal is to load training data
each example contains game state, policy and value
the calss will work with Dataloader in Pytorch to suply data to the model'''
import torch
from torch.utils.data import Dataset

class GameDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    '''getting the index and returns the match sample when each '''
    def __getitem__(self, idx):
        state, policy, value = self.data[idx]
        return torch.tensor(state, dtype=torch.float32), \
               torch.tensor(policy, dtype=torch.float32), \
               torch.tensor(value, dtype=torch.float32)