import torch
import torch.nn as nn
from typing import TypeVar, Generic
from torch.utils.data import DataLoader
from genetic_algorithm.genotype import NetworkGenotype
from model.genetic_network import GeneticNetwork

T = TypeVar('T', GeneticNetwork, GeneticNetwork)

class FitnessEvaluator():
    def eval_fitness(self, genotype: NetworkGenotype) -> float:
        raise NotImplementedError()
    
class DatasetFitnessEvaluator():
    def __init__(self, dataset, batch_size=128, shuffle=True, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
    def eval_fitness(self, model_type: T, genotype: NetworkGenotype):
        fitness = 0.
        model = model_type(genotype)
        model.eval()
        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, 
                                shuffle=self.shuffle, num_workers=self.num_workers)
        for inputs, labels in dataloader:
            predicts = model(inputs)
            predicts = torch.argmax(predicts, dim=1)
            fitness += (predicts == labels).sum()
        return fitness / len(self.dataset)
    