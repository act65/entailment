import unittest

import tensorflow as tf

import data
import treenn
import cts_satisfiability as csat

import led_parser

def fetch_data(batch_size):
    # fetch a single batch
    fname = '../logical_entailment_dataset/data/train.txt'
    return next(data.batch_data(data.read_data(fname), batch_size))

class Test(unittest.TestCase):
    def setUp(self):
        self.d_world = 10
        self.n_ops = 32
        self.d_embed = 64
        self.w = tf.random_normal([1, self.d_world])
        self.batch_size = 50

        self.parser = data.Parser(led_parser.propositional_language())

    def test_sat3cell_tree(self):
        sat3 = csat.Sat3Cell(self.d_world, self.n_ops)
        nn = treenn.TreeNN(sat3, self.parser)

        A, B, E = fetch_data(self.batch_size)

        y = nn(self.w, A)
        self.assertEqual(y.shape, [self.batch_size, self.n_ops])

if __name__ == '__main__':
    tf.enable_eager_execution()
    unittest.main()
