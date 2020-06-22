import torch
import torch.nn as nn
from model.genetic_module import GeneticNetwork

class SimpleCNN(GeneticNetwork):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5))
        self.fc1 = nn.Linear(32 * 12**2, 128)
        self.fc2 = nn.Linear(128, 10)
        
    @torch.no_grad()
    def forward(self, x):
        out = self.conv1(x)
        out = torch.relu(out)
        out = torch.max_pool2d(out, (2,2))
        out = torch.dropout(out, 0.2)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        
        return out
    
    @classmethod
    def genetic_schema(cls):
        schema = {
            'conv1': (1, 32, 5),
            'fc1': (4608, 128),
            'fc2': (128, 10)
        }
        return schema