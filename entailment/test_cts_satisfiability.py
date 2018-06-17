import unittest

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

import cts_satisfiability as csat
import data

class TestCtsSat(unittest.TestCase):
    def setUp(self):
        self.num_units = 30
        self.n_worlds= 24
        self.d_embed = 50
        self.batch_size = 10
        self.n_ops = 32 # len(prop_parser.language.symbols)

    def test_output_shape(self):
        sat3 = csat.Sat3Cell(self.n_ops, self.num_units, self.batch_size, self.n_worlds)
        w = tf.random_normal([self.n_worlds, self.num_units])

        # create fake batch and state
        batch_ids = list(range(self.batch_size))

        seq_len = 20
        n_past = 12
        states = [tf.random_normal((self.batch_size, self.num_units, self.n_worlds))
                  for i in range(n_past)]
        # NOTE this only tests the unary fns...
        args = [[np.random.randint(0, n_past)]
                for _ in range(n_past)]
        op_ids = np.random.randint(0, self.n_ops, self.batch_size)

        locs_n_ops_n_args = list(zip(range(n_past), op_ids, args))

        h = sat3(states, locs_n_ops_n_args, w)
        self.assertEqual(h.shape, (self.batch_size, self.num_units, self.n_worlds))

    def test_variable_existience(self):
        sat3 = csat.Sat3Cell(self.n_ops, self.num_units, self.batch_size, self.n_worlds)
        names = [var.name for var in sat3.variables]
        self.assertTrue('nullary/embeddings:0' in names)

        self.assertTrue('unary/b4:0' in names)
        self.assertTrue('unary/W4:0' in names)

        self.assertTrue('binary/b4:0' in names)
        self.assertTrue('binary/W4:0' in names)

if __name__ == '__main__':
    unittest.main()
