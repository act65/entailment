from __future__ import print_function


import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import led_parser

import possible_worlds_net as pwn
import data
import cts_satisfiability as csat
import treenn

def argumentparser():
    parser = argparse.ArgumentParser(description='Train a PWN')
    parser.add_argument('--num_units', type=int, default=64,
                        help='Dimension of each world')
    parser.add_argument('--n_worlds', type=int, default=64,
                        help='Number of worlds')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size...')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--logdir', type=str, default='/tmp/pwn/',
                        help='location to save logs')
    parser.add_argument('--datadir', type=str, default'../logical_entailment_dataset/data',
                        help='location of the data')
    return parser.parse_args()

def cross_entropy(p, t):
    t = tf.constant(t, dtype=tf.float32, shape=(p.shape[0], 1))
    with tf.name_scope('cross_entropy'):
        return tf.reduce_mean(-t*tf.log(p) - (1-t)*tf.log(1-p))

def accuracy(y, t):
    return np.mean(np.equal(np.round(y), np.array(t)))

def compute_step(model, A, B, t):
    """
    Run the model and collect the operation used to compute the loss,
    propagate the gradient of loss back through the ops.
    """
    with tf.GradientTape() as tape:
        y = model(A, B)
        loss = cross_entropy(y, t)
    grads = tape.gradient(loss, model.variables)

    for g, v in zip(grads, model.variables):
        if g is None:
            raise ValueError('No gradient for {}'.format(v.name))

    return loss, grads, y

def main(args):
    language = led_parser.propositional_language()
    parser = data.Parser(language)
    n_ops = len(language.symbols)

    # construct a pwn using a treenn with a sat3 cell
    sat3 = csat.Sat3Cell(n_ops, args.num_units, args.batch_size, args.n_worlds)
    nn = treenn.TreeNN(sat3, parser, args.batch_size)
    possibleworldsnet = pwn.PossibleWorlds(nn, args.n_worlds, args.num_units)

    print('N variables = {}'.format(np.sum([np.prod(var.shape)
                                    for var in possibleworldsnet.variables])))

    opt = tf.train.AdamOptimizer()
    writer = tf.contrib.summary.create_file_writer(args.logdir)
    writer.set_as_default()

    for e in range(args.epochs):
        # Train
        for A, B, E in data.fetch_data(args.batch_size):
            loss, grads, p = compute_step(possibleworldsnet, A, B, E)
            gnvs = zip(grads, possibleworldsnet.variables)
            step = tf.train.get_or_create_global_step()
            opt.apply_gradients(gnvs, global_step=step)

            print('\rstep: {} loss {:.4f}'.format(step.numpy(),
                                tf.reduce_mean(loss)), end='', flush=True)

            with tf.contrib.summary.record_summaries_every_n_global_steps(10):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('acc', accuracy(p, E))

        # Evaluate
        for test_name, test_set in data.fetch_test_sets('../logical_entailment_dataset/data',
                                        args.batch_size):
            print('\rEvaluating: {}'.format(test_name), end='', flush=True)
            acc = np.mean([accuracy(possibleworldsnet(A, B), E)
                           for A, B, E in test_set])
            tf.contrib.summary.scalar(test_name, acc)

if __name__ == "__main__":
    tf.enable_eager_execution()
    main(argumentparser())
