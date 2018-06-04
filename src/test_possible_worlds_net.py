import unittest

import tensorflow as tf

import led_parser

import possible_worlds_net as pwn
import data
import cts_satisfiability as csat
import treenn

class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.d_world = 30
        self.n_worlds=24
        self.d_embed = 50
        self.batch_size = 10
        self.n_ops = 32 # len(prop_parser.language.symbols)

    def test_simple_cell_output_shape(self):
        cell = lambda *x: x
        # pwn()

class TestIntegration(unittest.TestCase):
    def test_sat3_output_shape(self):
        """integration test with sat3"""
        d_world = 10
        n_worlds = 2
        n_ops = 32
        d_embed = 8
        batch_size = 100

        parser = data.Parser(led_parser.propositional_language())
        sat3 = csat.Sat3Cell(d_world, n_ops, batch_size)
        nn = treenn.TreeNN(sat3, parser, batch_size)
        possibleworldsnet = pwn.PossibleWorlds(nn, n_worlds, d_world)

        A, B, E = data.fetch_data(batch_size)
        y = possibleworldsnet(A, B)
        print(y)


if __name__ == '__main__':
    unittest.main()
