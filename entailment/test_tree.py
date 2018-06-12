import unittest

import tensorflow as tf

import data
import treenn
import cts_satisfiability as csat

import led_parser

class Test(unittest.TestCase):
    def setUp(self):
        self.d_world = 10
        self.n_ops = 32
        self.d_embed = 64
        self.batch_size = 50
        self.n_worlds = 64

        self.parser = data.Parser(led_parser.propositional_language())

    def test_sat3cell_tree(self):
        w = tf.random_normal([1, self.d_world])

        sat3 = csat.Sat3Cell(self.d_world, self.n_ops, self.batch_size, self.n_worlds)
        nn = treenn.TreeNN(sat3, self.parser, self.batch_size)

        A, B, E = next(data.fetch_data(self.batch_size))

        y = nn(w, [nn.parser(a) for a in A])
        self.assertEqual(y.shape, [self.batch_size, self.n_ops])

if __name__ == '__main__':
    unittest.main()
