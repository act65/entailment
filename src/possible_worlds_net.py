import tensorflow as tf

class PossibleWorlds():
    """
    A NN designed specifically for predicting entailment.
    """
    def __init__(self, encoder, n_worlds, d_world):
        self.encoder = encoder
        self.n_worlds = n_worlds
        self.worlds = tf.get_variable(shape=(n_worlds, d_world), dtype=tf.float32, name='worlds')

        self.dense = tf.keras.layers.Dense(1)

        # self.variables = (self.dense.variables +
        #                  [self.encoder.cell.b4,
        #                  self.encoder.cell.op_embeddings,
        #                  self.encoder.cell.W4])

    def eval_world(self, w):
        x = tf.concat([self.encoder(w, self.a_trees),
                     self.encoder(w, self.b_trees)], axis=1)
        return self.dense(x) # NOTE in the paper this isnt actually a dense layer....

    def __call__(self, A, B):
        """
        Convolve over possible worlds.
        """
        # parse each sent independently
        self.a_trees = [self.encoder.parser(a) for a in A]
        self.b_trees = [self.encoder.parser(b) for b in B]

        # self.eval_world(self.worlds[0, :])  # attempt to initialise the variables first

        # TODO if we could make this parallel it would be much faster!!
        # but require a large memory foot print!?
        # y = tf.map_fn(self.eval_world, self.worlds)
        # print(y)

        p = tf.constant(1.0, dtype=tf.float32)
        for i in range(self.n_worlds):
            p *= self.eval_world(self.worlds[i:i+1])
        return p
