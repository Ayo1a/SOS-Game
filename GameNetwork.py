import torch
import torch.nn as nn
import torch.nn.functional as F

class GameNetwork(nn.Module):
    def __init__(self):
        super(GameNetwork, self).__init__()
        # Convolutional layers with batch normalization
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU()
        )
        # Policy and value heads
        self.policy_head = nn.Linear(256, 8 * 8 * 2)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten for FC layers
        x = self.fc_layers(x)

        # Policy and value outputs
        policy = F.softmax(self.policy_head(x), dim=-1).view(-1, 8, 8, 2)
        value = torch.tanh(self.value_head(x))

        return policy, value

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))