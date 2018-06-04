import tensorflow as tf
import nary_fns

class Sat3Cell():
    def __init__(self, n_ops, num_units):
        self.num_units = num_units
        self.batch_size = 50

        self.nullary = nary_fns.Nullary(n_ops, num_units)
        self.unary = nary_fns.Unary(n_ops, num_units)
        self.binary = nary_fns.Binary(n_ops, num_units)

    def __call__(self, states, opsnargs):
        """
        Args:
            states (list): a list of tf.tensors of shape [?, ?]
            opsnargs (list):
        """
        ops, args = list(zip(*opsnargs))

        # bundle into nullary, unary and binary ops
        # so we can batch them together
        nullary_ops = []
        unary_ops = []
        binary_ops = []
        for j in range(len(opsnargs)):
            op, arg = opsnargs[j]
            if len(arg) == 0:
                nullary_ops.append((j, op))
            elif len(arg) == 1:
                unary_ops.append((j, op, arg))
            elif len(arg) == 2:
                binary_ops.append((j, op, arg))
            else:
                raise ValueError('Too many args. Got {}, but should '
                                 'be in [0,1,2]'.format(len(arg)) )

        state = tf.zeros([self.batch_size, self.num_units], dtype=tf.float32)
        # apply nullary ops as a batch
        if len(nullary_ops) > 0:
            state += self.nullary(nullary_ops)
        # apply unary ops as a batch
        if len(unary_ops) > 0:
            state += self.unary(states, unary_ops)
        # apply binary ops as a batch
        if len(binary_ops) > 0:
            state += self.binary(states, binary_ops)
        return state
