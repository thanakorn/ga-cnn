import torch
import torch.nn as nn
from genetic_algorithm.chromosomes import ChromosomeSchema

class GeneticNetwork(nn.Module):    
    @classmethod
    def genetic_schema(cls) -> ChromosomeSchema:
        raise NotImplementedError()