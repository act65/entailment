import unittest

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np

from nary_fns import Nullary, Unary, Binary
import data

class TestNullary(unittest.TestCase):
    def setUp(self):
        self.d_world = 30
        self.n_worlds=24
        self.d_embed = 30
        self.batch_size = 50
        self.n_ops = 32
        self.num_units = 10

    def test_single_world(self):
        null = Nullary(self.n_ops, self.num_units, self.batch_size)

        bundle_size = 15
        bundle_ids = np.sort(np.random.randint(0, self.batch_size, bundle_size))
        op_ids = np.random.randint(0, self.n_ops, bundle_size)
        x = np.stack([bundle_ids, op_ids], axis=1)

        w = tf.random_normal(shape=(self.n_worlds, self.num_units))

        y = null(x, w[0:1, ...])
        self.assertEqual(y.shape, (self.batch_size, self.num_units, 1))


    def test_output_shape(self):
        null = Nullary(self.n_ops, self.num_units, self.batch_size)

        bundle_size = 15
        bundle_ids = np.sort(np.random.randint(0, self.batch_size, bundle_size))
        op_ids = np.random.randint(0, self.n_ops, bundle_size)
        x = np.stack([bundle_ids, op_ids], axis=1)

        w = tf.random_normal(shape=(self.n_worlds, self.num_units))

        y = null(x, w)
        self.assertEqual(y.shape, (self.batch_size, self.num_units, self.n_worlds))


class TestUnary(unittest.TestCase):
    def setUp(self):
        self.d_world = 30
        self.n_worlds=24
        self.d_embed = 30
        self.batch_size = 50
        self.n_ops = 32
        self.num_units = 10

    def test_many(self):
        unary = Unary(self.n_ops, self.num_units, self.batch_size)

        # create fake bundle and state
        bundle_size = 15
        bundle_ids = np.sort(np.random.randint(0, self.batch_size, bundle_size))

        n_past = 12
        states = [tf.random_normal((self.batch_size, self.num_units, self.n_worlds)) for i in range(n_past)]
        args = [[np.random.randint(0, n_past)]
                for _ in range(n_past)]
        op_ids = np.random.randint(0, self.n_ops, bundle_size)

        locs_n_ops_n_args = list(zip(range(n_past), op_ids, args))

        y = unary(states, locs_n_ops_n_args)
        print('Unary output: {}'.format(y.shape))
        self.assertEqual(y.get_shape().as_list(), [self.batch_size, self.num_units, self.n_worlds])

    def test_variables(self):
        binary = Unary(self.n_ops, self.num_units, self.batch_size)
        names = [v.name for v in binary.variables]
        for name in ['unary/W4:0', 'unary/b4:0']:
            self.assertTrue(name in names)

class TestBinary(unittest.TestCase):
    def setUp(self):
        self.d_world = 30
        self.n_worlds=24
        self.d_embed = 30
        self.batch_size = 50
        self.n_ops = 32
        self.num_units = 10

    def test_many(self):
        binary = Binary(self.n_ops, self.num_units, self.batch_size)

        # create fake bundle and state
        bundle_size = 15
        bundle_ids = np.sort(np.random.randint(0, self.batch_size, bundle_size))

        n_past = 12
        states = [tf.random_normal((self.batch_size, self.num_units, self.n_worlds)) for i in range(n_past)]
        args = [[np.random.randint(0, n_past), np.random.randint(0, n_past)]
                for _ in range(n_past)]
        op_ids = np.random.randint(0, self.n_ops, bundle_size)

        locs_n_ops_n_args = list(zip(range(n_past), op_ids, args))

        y = binary(states, locs_n_ops_n_args)
        print('Binary output: {}'.format(y.shape))
        self.assertEqual(y.get_shape().as_list(), [self.batch_size, self.num_units, self.n_worlds])

    def test_variables(self):
        binary = Binary(self.n_ops, self.num_units, self.batch_size)
        names = [v.name for v in binary.variables]
        for name in ['binary/W4:0', 'binary/b4:0']:
            self.assertTrue(name in names)

if __name__ == '__main__':
    unittest.main()
