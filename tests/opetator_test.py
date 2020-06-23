import unittest
import torch
import torch.nn as nn
from genetic_algorithm.genotype import NetworkGenotype
from genetic_algorithm.chromosomes import *
from genetic_algorithm.opetators import *

schema = {
    'conv': ConvChromosome(3, 16, 3, 4),
    'fc': LinearChromosome(128, 3)
}
    
class OpetatorTest(unittest.TestCase):
    
    def test_select_elites(self):
        genotypes = [NetworkGenotype(schema) for i in range(7)]
        fitnesses = [0.01, 1.0, 0.8, 0.2, 0.9, 0.75, 0.2]
        elites = select_elites(genotypes, fitnesses, 3)
        self.assertEqual(elites[0], genotypes[2])
        self.assertEqual(elites[1], genotypes[4])
        self.assertEqual(elites[2], genotypes[1])
    
    def test_mutation(self):
        parent = NetworkGenotype(schema)
        child = mutate(parent)
        self.assertFalse(torch.equal(parent.chromosomes['conv.weight'], child.chromosomes['conv.weight']))
        self.assertFalse(torch.equal(parent.chromosomes['conv.bias'], child.chromosomes['conv.weight']))
        self.assertFalse(torch.equal(parent.chromosomes['fc.weight'], child.chromosomes['fc.weight']))
        self.assertFalse(torch.equal(parent.chromosomes['fc.bias'], child.chromosomes['fc.weight']))