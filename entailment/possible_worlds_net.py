import tensorflow as tf

import copy

class PossibleWorlds():
    """
    A NN designed specifically for predicting entailment.
    """
    def __init__(self, encoder, n_worlds, d_world):
        self.encoder = encoder
        self.n_worlds = n_worlds

        with tf.variable_scope('pwn', reuse=tf.AUTO_REUSE):
            self.worlds = tf.get_variable(shape=(n_worlds, d_world),
                                          dtype=tf.float32,
                                          name='worlds')
            self.W = tf.get_variable(
                shape=[encoder.cell.num_units*2, 1],
                dtype=tf.float32,
                name='W',
                initializer=tf.random_normal_initializer(mean=1.0, stddev=0.1))
            self.b = tf.get_variable(
                shape=[1, 1],
                dtype=tf.float32,
                name='b',
                initializer=tf.constant_initializer(1))

        self.variables = [self.W, self.b, self.worlds] +  self.encoder.variables


    def eval_world(self, w):
        # NOTE this was a hard bug to catch. for some reason the trees are being
        # mutated. not sure what is doing this.
        x = tf.concat([self.encoder(w, copy.deepcopy(self.a_trees)),
                     self.encoder(w, copy.deepcopy(self.b_trees))], axis=1)
        # NOTE in the paper this isnt actually a dense layer...
        return tf.nn.sigmoid(tf.matmul(x, self.W) + self.b)

    def __call__(self, A, B):
        """
        Convolve over possible worlds.
        """
        # parse each sent independently
        # parse here instead of repeatedly within self.encoder
        # TODO use a learned parser...
        self.a_trees = [self.encoder.parser(a) for a in A]
        self.b_trees = [self.encoder.parser(b) for b in B]

        # TODO if we could make this parallel it would be much faster!!
        # but require a large memory foot print!?
        # self.eval_world(self.worlds[0, :])  # attempt to initialise the variables first

        # y = tf.map_fn(self.eval_world, self.worlds)
        # print(y)

        p = tf.constant(1.0, dtype=tf.float32)
        # NOTE this probably leads to poor inits?
        for i in range(self.n_worlds):
            p *= self.eval_world(self.worlds[i:i+1])
        return p
