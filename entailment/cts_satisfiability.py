import tensorflow as tf
import nary_fns

class Sat3Cell():
    def __init__(self, n_ops, num_units, batch_size, n_worlds):
        self.num_units = num_units
        self.batch_size = batch_size
        self.n_ops = n_ops
        self.n_worlds = n_worlds

        self.nullary = nary_fns.Nullary(n_ops, num_units, batch_size)
        self.unary = nary_fns.Unary(n_ops, num_units, batch_size)
        self.binary = nary_fns.Binary(n_ops, num_units, batch_size)

        self.variables = (self.nullary.variables +
                          self.unary.variables +
                          self.binary.variables)

    def __call__(self, states, locs_n_ops_n_args, w):
        """
        Args:
            states (list): a list of tf.tensors of shape [?, ?]
            opsnargs (list):
        """
        locs, ops, args = list(zip(*locs_n_ops_n_args))
        n = len(locs)

        # bundle into nullary, unary and binary ops
        # so we can batch them together
        nullary_ops = []
        unary_ops = []
        binary_ops = []
        for j in range(n):
            loc, op, arg = locs_n_ops_n_args[j]

            if len(arg) == 0:
                nullary_ops.append((loc, op))
            elif len(arg) == 1:
                unary_ops.append((loc, op, arg))
            elif len(arg) == 2:
                binary_ops.append((loc, op, arg))
            else:
                raise ValueError('Too many args. Got {}, but should '
                                 'be in [0,1,2]'.format(len(arg)) )

        state = tf.zeros([self.batch_size, self.num_units, self.n_worlds], dtype=tf.float32)
        # apply nullary ops as a batch
        if len(nullary_ops) > 0:
            state += self.nullary(nullary_ops, w)
        # apply unary ops as a batch
        if len(unary_ops) > 0:
            state += self.unary(states, unary_ops)
        # apply binary ops as a batch
        if len(binary_ops) > 0:
            state += self.binary(states, binary_ops)
        return state
