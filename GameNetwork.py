import torch
import torch.nn as nn
import torch.nn.functional as F

class GameNetwork(nn.Module):
    def __init__(self, board_size=5):  # שינוי לגודל לוח חדש
        super(GameNetwork, self).__init__()
        self.board_size = board_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.residual_block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )

        # עדכון שכבות Fully Connected בהתאם לגודל החדש
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # ראש המדיניות - שינוי גודל הפלט בהתאם ללוח 5x5
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size * 2),
            nn.Softmax(dim=-1)
        )

        # ראש הערך - שינוי גודל הפלט בהתאם ללוח 5x5
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x + self.residual_block(x)  # חיבור רזידואלי
        x = F.relu(x)

        policy = self.policy_head(x).view(-1, self.board_size, self.board_size, 2)
        value = self.value_head(x)

        return policy, value

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
