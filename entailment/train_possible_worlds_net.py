import argparse
import tensorflow as tf

import led_parser

import possible_worlds_net as pwn
import data
import cts_satisfiability as csat
import treenn

def cross_entropy(p, t):
    t = tf.constant(t, dtype=tf.float32, shape=(p.shape[0], 1))
    with tf.name_scope('cross_entropy'):
        return tf.reduce_mean(-t*tf.log(p) - (1-t)*tf.log(1-p))

def compute_gradients(model, A, B, t):
    with tf.GradientTape() as tape:
        y = model(A, B)
        loss = cross_entropy(y, t)
        step = tf.train.get_or_create_global_step().numpy()
        print('\rstep: {} loss {}'.format(step, tf.reduce_mean(loss)), end='', flush=True)
    return tape.gradient(loss, model.variables)

def main():
    language = led_parser.propositional_language()
    parser = data.Parser(language)

    # TODO explore how the speed scales with these parameters
    d_world = 10
    n_worlds = 16
    n_ops = len(language.symbols)
    d_embed = 8
    batch_size = 50

    sat3 = csat.Sat3Cell(n_ops, d_world, batch_size)
    nn = treenn.TreeNN(sat3, parser, batch_size)
    possibleworldsnet = pwn.PossibleWorlds(nn, n_worlds, d_world)

    opt = tf.train.AdamOptimizer()

    for A, B, E in data.fetch_data(batch_size):
        grads = compute_gradients(possibleworldsnet, A, B, E)
        gnvs = zip(grads, possibleworldsnet.variables)
        opt.apply_gradients(gnvs, global_step=tf.train.get_or_create_global_step())

if __name__ == "__main__":
    tf.enable_eager_execution()
    main()
