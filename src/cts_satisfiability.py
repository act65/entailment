import tensorflow as tf
import numpy as np

def scatter_add(ref, idx, x):
    if tf.executing_eagerly():
        ref = tf.contrib.eager.Variable(ref)
    else:
        ref = tf.Variable(ref)
    return tf.scatter_add(ref, idx, x)

class Nullary():
    def __init__(self, n_ops, num_units, batch_size=50):
        self.n_ops = n_ops
        self.num_units = num_units
        self.batch_size = batch_size
        self.embed = tf.keras.layers.Embedding(n_ops, num_units)

    def __call__(self, x):
        """
        Args:
            x (tf.tensor): shape [bundle_size, 2].
                dim 0: the ids of the batch elems
                dim 1: the ops to embed
        Returns:
            (tf.tensor): [batch_size, self.num_units]
        """
        y = tf.zeros([self.batch_size, self.num_units], dtype=tf.float32)
        x = tf.constant(x, dtype=tf.int32)
        e = self.embed(x[:, 1])  # fetch embeddings

        return scatter_add(y, x[:,0], e)

class Binary():
    def __init__(self, n_ops, num_units, batch_size=50):
        self.num_units = num_units
        self.batch_size = batch_size
        self.W4 = tf.get_variable(shape=(n_ops, 2*num_units, num_units), dtype=tf.float32, name='W4')
        self.b4 = tf.get_variable(shape=(n_ops, num_units), dtype=tf.float32, name='b4')

    def __call__(self, states, opsnargs):
        """
        Simple fn made complicated by wanting to dynamically batch ops together.

        ```
        def fn(op, l, r):
            x = concat(l,r)
            y = matmul(W_op, x) + b_op
            return normalize(y)
        ```

        Args:
            state (list): the previously calculated nodes in depth first order
                shape in []
            opsnargs ():
            locs (list): the trees to that the ops args apply to
        """
        locs, ops, args = list(zip(*opsnargs))
        n_bundle = len(locs)

        ops = tf.constant(ops, dtype=tf.int32)
        args = np.array(args)
        stack = tf.concat(states, axis=0)

        # since we have stacked the batches and past states
        # we need to index like so
        l_idx = args[:, 0]*self.batch_size + locs
        r_idx = args[:, 1]*self.batch_size + locs

        l = tf.gather(stack, l_idx)
        r = tf.gather(stack, r_idx)

        # the actual cell ...
        x = tf.concat([l, r], axis=1)  # shape [n_bundle x 2.num_units]

        h = [] # tf.zeros(shape=[n_bundle, self.num_units], dtype=tf.float32)
        W = tf.gather(self.W4, ops)  # shape [n_bundle x 2.num_units x num_units]

        # can decompose matmul as a sum of products
        for i in range(self.num_units):
            # (batch_size x 2.num_units) . (2.num_units x num_units)
            h.append(x*W[:, :, i])  # products
        h = tf.reduce_sum(tf.stack(h, axis=-1), axis=1)  # sum

        h = tf.nn.l2_normalize(h, axis=1)

        y = tf.zeros([self.batch_size, self.num_units], dtype=tf.float32)

        return scatter_add(y, locs, h)

# class Unary():
#     def __init__(self):
#         pass

class Sat3Cell():
    def __init__(self, n_ops, num_units):
        self.num_units = num_units
        self.batch_size = 50

        self.nullary = Nullary(n_ops, num_units)
        self.binary = Binary(n_ops, num_units)

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
        binary_ops = []
        for j in range(len(opsnargs)):
            op, arg = opsnargs[j]
            if len(arg) == 0:
                nullary_ops.append((j, op))
            elif len(arg) == 1:
                pass
            elif len(arg) == 2:
                binary_ops.append((j, op, arg))
            else:
                raise ValueError('Too many args. Got {}, but should '
                                 'be in [0,1,2]'.format(len(arg)) )

        # apply nullary ops as a batch
        state = tf.zeros([self.batch_size, self.num_units], dtype=tf.float32)
        if len(nullary_ops) > 0:
            state += self.nullary(nullary_ops)
        # apply binary ops as a batch
        if len(binary_ops) > 0:
            state += self.binary(states, binary_ops)
        return state
