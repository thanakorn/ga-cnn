import torch
import torch.nn as nn
from model.genetic_network import GeneticNetwork
from genetic_algorithm.chromosomes import *

class SimpleCNN(GeneticNetwork):
    @classmethod
    def genetic_schema(cls):
        schema = {
            'conv1': ConvChromosome(1, 32, 5, 1),
            'fc1': LinearChromosome(4608, 128),
            'fc2': LinearChromosome(128, 10)
        }
        return schema
    
    @torch.no_grad()
    def forward(self, x):
        out = self.conv1(x)
        out = torch.relu(out)
        out = torch.max_pool2d(out, (2,2))
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        
        return out