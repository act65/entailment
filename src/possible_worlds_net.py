import tensorflow as tf

class PossibleWorlds():
  """
  A NN designed specifically for predicting entailment.
  """
  def __init__(self, encoder, num_units, n_worlds, d_world):
    self.encoder = encoder
    self.n_worlds = n_worlds
    self.worlds = tf.get_variable(shape=(n_worlds, d_world), dtype=tf.float32, name='worlds')

    self.dense = tf.keras.layers.Dense(num_units)

    self.variables = (self.dense.variables +
                     [self.encoder.cell.b4,
                     self.encoder.cell.op_embeddings,
                     self.encoder.cell.W4])

  def inner(self, a, b):
    """
    Convolve over possible worlds.
    For each random direction, do !??!

    """
    p = tf.constant(1.0, dtype=tf.float32)
    for i in range(self.n_worlds):

      x = tf.concat([self.encoder(self.worlds[i:i+1], a),
                     self.encoder(self.worlds[i:i+1], b)], axis=1)
      p *= self.dense(x)  # in the paper this isnt actually a dense layer....

    return p

  def __call__(self, A, B):
    """
    For each element of a batch.
    """
    return tf.concat([self.inner(a, b) for a, b in zip(A, B)], axis=0)
