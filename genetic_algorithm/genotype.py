import torch

from logging import Logger
from genetic_algorithm.chromosomes import ChromosomeSchema, ConvChromosome, LinearChromosome

class NetworkGenotype():
    def __init__(self, schema: ChromosomeSchema):
        self.schema = schema
        self.chromosomes = {}
        for name, chromosome in schema.items():
            if isinstance(chromosome, ConvChromosome):
                in_channel, out_channel, kernel_size = chromosome
                self.chromosomes[name] = torch.rand((out_channel, in_channel, kernel_size, kernel_size))
            elif isinstance(chromosome, LinearChromosome):
                in_feature, out_feature = chromosome
                self.chromosomes[name] = torch.rand((out_feature, in_feature))
                