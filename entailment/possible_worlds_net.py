import tensorflow as tf

import copy

class PossibleWorlds():
    """
    A NN designed specifically for predicting entailment.
    See https://arxiv.org/abs/1802.08535 for details.
    """
    def __init__(self, encoder, n_worlds, d_world):
        """
        Args:
            encoder (f: world x tree -> probability)
            n_worlds (int): the number of worlds to use
            d_world (int): the dimensionality of those worlds
        """
        self.encoder = encoder
        self.n_worlds = n_worlds
        self.d_world = d_world

        with tf.variable_scope('pwn', reuse=tf.AUTO_REUSE):
            self.worlds = tf.get_variable(shape=(n_worlds, d_world),
                                          dtype=tf.float32,
                                          name='worlds')
            self.W = tf.get_variable(
                shape=[encoder.cell.num_units*2, 1],
                dtype=tf.float32,
                name='W',
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
            self.b = tf.get_variable(
                shape=[1, 1],
                dtype=tf.float32,
                name='b',
                # NOTE set init to 3 bc we want to help p be equal to 1
                initializer=tf.constant_initializer(3))

        self.variables = [self.W, self.b, self.worlds] +  self.encoder.variables


    def eval_world(self, a, b, w):
        """
        Args:
            a (list): a batch of trees
            b (list): a batch of trees
            w (tf.tensor): shape [1, d_world], dtype tf.float32

        Returns:
            p (tf.tensor): probability that a entails b. shape [batch_size, 1]
        """
        # NOTE this was a hard bug to catch. for some reason the trees are being
        # mutated. not sure what is doing this.
        x = tf.concat([self.encoder(w, copy.deepcopy(a)),
                       self.encoder(w, copy.deepcopy(b))], axis=1)
        # NOTE in the paper this isnt actually a dense layer...
        # [batch_size x num_units*2 x n_worlds] x [num_units*2 x 1]
        y = tf.layers.flatten(tf.tensordot(x, self.W, axes=[[1], [0]])) + self.b
        return tf.nn.sigmoid(y)

    def __call__(self, A, B):
        """
        Convolve over possible worlds.
        """
        # parse each sent independently
        # parse here instead of repeatedly within self.encoder
        # TODO use a learned parser...??
        a_trees = [self.encoder.parser(a) for a in A]
        b_trees = [self.encoder.parser(b) for b in B]

        p = self.eval_world(a_trees, b_trees, self.worlds)
        return tf.foldl(tf.multiply, tf.unstack(p, axis=1))
