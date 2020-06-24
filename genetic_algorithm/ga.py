import torch
import copy
import numpy as np
import concurrent.futures

from concurrent.futures import as_completed
from tqdm import tqdm, trange
from typing import List
from genetic_algorithm.genotype import NetworkGenotype
from genetic_algorithm.opetators import *
from model.genetic_network import GeneticNetwork
from typing import TypeVar

M = TypeVar('M', GeneticNetwork, GeneticNetwork)

class GeneticAlgorithm():
    def __init__(self, model_type: M, num_population, fitness_evaluator, selection_pressure=0.1, mutation_rate=0.01, crossover_rate=0.01):
        self.model_type = model_type
        schema = model_type.genetic_schema()
        self.num_populations = num_population
        self.populations = [NetworkGenotype(schema) for i in range(num_population)]
        self.fitness_evaluator = fitness_evaluator
        self.selection_pressure = selection_pressure
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
    def run(self, num_generations):
        populations = []
        best = None
        elites = []
        fitnesses = np.zeros(self.num_populations)
        num_elites = int(self.selection_pressure * self.num_populations)
        for gen in range(num_generations):
            # Generate population
            if gen == 0:
                populations = self.populations
            else:
                populations = gen_population_mutation(elites, n=self.num_populations - 1, mutation_power=1.0)
                populations.append(best)
            
            # Evaluate fitness concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor, tqdm(total=len(populations), desc=f'Generation {gen+1}') as t:
                fitness_futures = [executor.submit(self.fitness_evaluator.eval_fitness, self.model_type, p) for p in populations]
                for i, f in zip(range(self.num_populations), as_completed(fitness_futures)):
                    fitnesses[i] = f.result()
                    t.update()
                t.set_postfix(max_f=np.max(fitnesses), min_f=np.min(fitnesses), avg_f=np.average(fitnesses))

            # Choose elites
            elites = select_elites(populations, fitnesses, num_elites)
            best = elites[-1]
            
        return best
                
    
            
            