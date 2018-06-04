import unittest

import tensorflow as tf

import possible_worlds_net as pwn

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
        pass


if __name__ == '__main__':
    unittest.main()
