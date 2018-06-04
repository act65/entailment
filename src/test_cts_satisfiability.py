import unittest

import tensorflow as tf

import cts_satisfiability as csat

class TestCtsSat(unittest.TestCase):
    def setUp(self):
        self.d_world = 30
        self.n_worlds=24
        self.d_embed = 50
        self.batch_size = 10
        self.n_ops = 32 # len(prop_parser.language.symbols)

    def test_output_shape(self):
        sat3 = csat.Sat3Cell(self.d_world, self.n_ops)
        w = tf.random_normal([1, self.d_world])

        h = sat3(w, 0)
        self.assertEqual(h.shape, (1, self.d_embed))

        h = sat3(w, 12, l, r)
        self.assertEqual(h.shape, (1, self.d_embed))


if __name__ == '__main__':
    unittest.main()
