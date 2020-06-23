import torch
from genetic_algorithm.genotype import NetworkGenotype
from genetic_algorithm.chromosomes import *

def mutate(genotype: NetworkGenotype, mutation_power=0.01) -> NetworkGenotype:
    child = genotype.clone()
    for _, chromosome in child.chromosomes.items():
        chromosome += mutation_power * torch.randn(chromosome.shape)
    return child
        