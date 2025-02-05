import torch
import torch.nn as nn

class GameNetwork(nn.Module):
    def __init__(self, input_size, action_size):
        super(GameNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, action_size) #convert results to probabilities vector 
        self.value_head = nn.Linear(128, 1) #valuation for curent stat 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = torch.softmax(self.policy_head(x), dim=-1) 
        value = torch.tanh(self.value_head(x)) #evaluation between -1 to 1 
        return policy, value

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))