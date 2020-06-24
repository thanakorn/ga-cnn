import torch
import numpy as np

from torch import Tensor
from typing import List
from genetic_algorithm.genotype import NetworkGenotype
from genetic_algorithm.chromosomes import *

def select_elites(populations: List[NetworkGenotype], fitnesses, n) -> List[NetworkGenotype]:
    elite_idx = np.argsort(fitnesses)[-n:]
    elites = [populations[i] for i in elite_idx]
    return elites

def mutate(genotype: NetworkGenotype, mutation_power) -> NetworkGenotype:
    child = genotype.clone()
    for _, chromosome in child.chromosomes.items():
        chromosome += mutation_power * torch.randn(chromosome.shape)
    return child

def crossover(a: NetworkGenotype, b:NetworkGenotype) -> NetworkGenotype:
    c = a.clone()
    for name in a.chromosomes.keys():
        a_chromosome, b_chromosome = a.chromosomes[name], b.chromosomes[name]
        c.chromosomes[name] = a_chromosome.clone() if np.random.rand() <= 0.5 else b_chromosome.clone()
    return c

def gen_population_mutation(parents: List[NetworkGenotype], n, mutation_power=0.01):
    new_generation = []
    K = np.random.randint(0, len(parents), n)
    parents = [parents[k] for k in K]
    for p in parents: new_generation.append(mutate(p, mutation_power))
    return new_generation