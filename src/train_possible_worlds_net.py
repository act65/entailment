import tensorflow as tf

def compute_gradients(model, x, t):
    with tf.GradientTape() as tape:
        y = model(x)
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=t,logits=y)
        step = tf.train.get_or_create_global_step().numpy()
        print('\rstep: {} loss {}'.format(step, tf.reduce_mean(loss)), end='', flush=True)

  return tape.gradient(loss, variables)

def main():
    possibleworldsnet = PossibleWorlds(
        encoder=TreeNN(Sat3Cell(d_world, n_ops, d_embed), prop_parser),
        num_units=1,
        n_worlds=n_worlds,
        d_world=d_world
    )

    opt = tf.train.AdamOptimizer()

    for A, B, E in batch_data(data, batch_size):
        E = tf.constant(t, dtype=tf.float32, shape=(batch_size, 1))
        grads = compute_gradients(model=possibleworldsnet, x=(A, B), t=E)
        gnvs = zip(grads, possibleworldsnet.variables)
        opt.apply_gradients(gnvs, global_step=tf.train.get_or_create_global_step())

if __name__ == "__main__":
    tf.enable_eager_execution()
    main()
