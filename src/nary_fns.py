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

class Unary():
    def __init__(self, n_ops, num_units, batch_size=50):
        self.num_units = num_units
        self.batch_size = batch_size
        with tf.variable_scope('unary'):
            self.W4 = tf.get_variable(shape=(n_ops, num_units, num_units), dtype=tf.float32, name='W4')
            self.b4 = tf.get_variable(shape=(n_ops, num_units), dtype=tf.float32, name='b4')

    def __call__(self, states, opsnargs):
        """
        Simple fn made complicated by wanting to dynamically batch ops together.

        ```
        def fn(op, l):
            y = matmul(W_op, l) + b_op
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

        l = tf.gather(stack, l_idx)

        # the actual cell ...
        x = l  # shape [n_bundle x num_units]

        h = [] # tf.zeros(shape=[n_bundle, self.num_units], dtype=tf.float32)
        W = tf.gather(self.W4, ops)  # shape [n_bundle x num_units x num_units]

        # can decompose matmul as a sum of products
        for i in range(self.num_units):
            # (batch_size x num_units) . (num_units x num_units)
            h.append(x*W[:, :, i])  # products
        h = tf.reduce_sum(tf.stack(h, axis=-1), axis=1)  # sum

        h = tf.nn.l2_normalize(h, axis=1)

        y = tf.zeros([self.batch_size, self.num_units], dtype=tf.float32)

        return scatter_add(y, locs, h)

class Binary():
    def __init__(self, n_ops, num_units, batch_size=50):
        self.num_units = num_units
        self.batch_size = batch_size

        with tf.variable_scope('binary'):
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
