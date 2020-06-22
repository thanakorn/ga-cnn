import unittest
import torch
from genetic_algorithm.chromosomes import ConvChromosome, LinearChromosome
from genetic_algorithm.genotype import NetworkGenotype

class NetworkGenotypeTest(unittest.TestCase):
    def test_network_genotype_create_parameters(self):
        network_schema = {
            'conv1': ConvChromosome(3,16,4),
            'conv2': ConvChromosome(16,32,5),
            'fc1': LinearChromosome(12,8),
            'fc2': LinearChromosome(8,2),
        }
        network_genotype = NetworkGenotype(network_schema)
        self.assertTrue('conv1.weight' in network_genotype.chromosomes)
        self.assertTrue('conv1.bias' in network_genotype.chromosomes)
        self.assertTrue('conv2.weight' in network_genotype.chromosomes)
        self.assertTrue('conv2.bias' in network_genotype.chromosomes)
        self.assertTrue('fc1.weight' in network_genotype.chromosomes)
        self.assertTrue('fc1.bias' in network_genotype.chromosomes)
        self.assertTrue('fc2.weight' in network_genotype.chromosomes)
        self.assertTrue('fc2.bias' in network_genotype.chromosomes)
        
    def test_network_genotype_parameter_dimension(self):
        network_schema = {
            'conv1': ConvChromosome(3,16,4),
            'fc1': LinearChromosome(12,2)
        }
        network_genotype = NetworkGenotype(network_schema)
        conv1_weight = network_genotype.chromosomes['conv1.weight']
        conv1_bias = network_genotype.chromosomes['conv1.bias']
        fc1_weight = network_genotype.chromosomes['fc1.weight']
        fc1_bias = network_genotype.chromosomes['fc1.bias']
        self.assertTrue(torch.Size([16, 3, 4, 4]) == conv1_weight.shape)
        self.assertTrue(torch.Size([16]) == conv1_bias.shape)
        self.assertTrue(torch.Size([2, 12]) == fc1_weight.shape)
        self.assertTrue(torch.Size([2]) == fc1_bias.shape)