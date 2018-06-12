import tensorflow as tf
import numpy as np

def scatter_add(idx, x, batch_size):
    # TODO. still think this could be done in a nicer way!?
    # must be a more parallel way to do this?
    # was using tf.scatter_add but that required new variables to be
    # instantiated which didnt allow grads to flow
    indexes = [True if i in idx else False
               for i in range(batch_size)]
    vals = []
    counter = 0
    for i in range(batch_size):
        if indexes[i]:  # if we are scattering into this idx
            vals.append(x[counter, ...])
            counter += 1
        else: # else just pad with zeros
            vals.append(tf.zeros(x.shape[1:]))

    return tf.stack(vals)

class Nullary():
    def __init__(self, n_ops, num_units, batch_size):
        self.n_ops = n_ops
        self.num_units = num_units
        self.batch_size = batch_size
        with tf.variable_scope('nullary', reuse=tf.AUTO_REUSE):
            self.embeddings = tf.get_variable(shape=[n_ops, num_units, num_units],
                                              dtype=tf.float32,
                                              initializer=tf.orthogonal_initializer(),
                                              name='embeddings')

        self.variables = [self.embeddings]

    def __call__(self, x, w):
        """
        Args:
            x (tf.tensor): shape [bundle_size, 2].
                dim 0: the ids of the batch elems
                dim 1: the ops to embed
            w (tf.tensor
        Returns:
            (tf.tensor): [batch_size, self.num_units]
        """
        x = tf.constant(x, dtype=tf.int32)

        # fetch op embeddings [n_bundle x num_units x num_units]
        e = tf.gather(self.embeddings, x[:, 1])
        # TODO could batch worlds up here!? W = [n_worlds, num_units]
        # just need to make sure the other layers can handle the extra dims
        h = tf.tensordot(e, w, axes=[[1], [1]])
        h = tf.nn.l2_normalize(h, axis=1)

        return scatter_add(list(x[:, 0].numpy()), h, self.batch_size)


class Unary():
    def __init__(self, n_ops, num_units, batch_size):
        self.num_units = num_units
        self.batch_size = batch_size
        with tf.variable_scope('unary', reuse=tf.AUTO_REUSE):
            self.W4 = tf.get_variable(shape=(n_ops, num_units, num_units),
                                      dtype=tf.float32,
                                      initializer=tf.orthogonal_initializer(),
                                      name='W4')
            self.b4 = tf.get_variable(shape=(n_ops, num_units),
                                      dtype=tf.float32,
                                      initializer=tf.orthogonal_initializer(),
                                      name='b4')

        self.variables = [self.W4, self.b4]

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
        idx = args[:, 0]*self.batch_size + locs

        x = tf.gather(stack, idx)

        W = tf.gather(self.W4, ops)  # shape [n_bundle x 2.num_units x num_units]
        b = tf.gather(self.b4, ops)

        h = tf.tensordot(W, x, axes=[[1], [1]])
        h = tf.reduce_mean(h, axis=2) + tf.expand_dims(b, -1)
        h = tf.nn.l2_normalize(h, axis=1)

        return scatter_add(locs, h, self.batch_size)

class Binary():
    def __init__(self, n_ops, num_units, batch_size):
        self.num_units = num_units
        self.batch_size = batch_size

        with tf.variable_scope('binary', reuse=tf.AUTO_REUSE):
            self.W4 = tf.get_variable(shape=(n_ops, 2*num_units, num_units),
                                      dtype=tf.float32,
                                      initializer=tf.orthogonal_initializer(),
                                      name='W4')
            self.b4 = tf.get_variable(shape=(n_ops, num_units),
                                      dtype=tf.float32,
                                      initializer=tf.orthogonal_initializer(),
                                      name='b4')

        self.variables = [self.W4, self.b4]


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

        W = tf.gather(self.W4, ops)  # shape [n_bundle x 2.num_units x num_units]
        b = tf.gather(self.b4, ops)

        h = tf.tensordot(W, x, axes=[[1], [1]])
        h = tf.reduce_mean(h, axis=2) + tf.expand_dims(b, -1)
        h = tf.nn.l2_normalize(h, axis=1)

        return scatter_add(locs, h, self.batch_size)
