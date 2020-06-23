import torch
import numpy as np
from typing import List
from genetic_algorithm.genotype import NetworkGenotype

def select_elites(populations: List[NetworkGenotype], fitnesses, n):
    elite_idx = np.argsort(fitnesses)[-n:]
    elites = [populations[i] for i in elite_idx]
    return elites

def mutate(genotype: NetworkGenotype, mutation_power=0.01) -> NetworkGenotype:
    child = genotype.clone()
    for _, chromosome in child.chromosomes.items():
        chromosome += mutation_power * torch.randn(chromosome.shape)
    return child