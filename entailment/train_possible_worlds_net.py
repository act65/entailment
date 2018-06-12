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
    parser.add_argument('--d_world', type=int, default=10,
                        help='Dimension of each world')
    parser.add_argument('--n_worlds', type=int, default=16,
                        help='Number of worlds')
    parser.add_argument('--d_embed', type=int, default=8,
                        help='Dimension of embedding for symbols')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size...')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs')
    return parser.parse_args()

def cross_entropy(p, t):
    t = tf.constant(t, dtype=tf.float32, shape=(p.shape[0], 1))
    with tf.name_scope('cross_entropy'):
        return tf.reduce_mean(-t*tf.log(p) - (1-t)*tf.log(1-p))

def compute_step(model, A, B, t):
    """
    Run the model and collect the operation used to compute the loss,
    propagate the gradient of loss back through the ops.
    """
    with tf.GradientTape() as tape:
        y = model(A, B)
        loss = cross_entropy(y, t)
    grads = tape.gradient(loss, model.variables)

    return loss, grads, y

def main(args):
    language = led_parser.propositional_language()
    parser = data.Parser(language)
    n_ops = len(language.symbols)

    # TODO explore how the speed scales with hparams

    sat3 = csat.Sat3Cell(n_ops, args.d_world, args.batch_size)
    nn = treenn.TreeNN(sat3, parser, args.batch_size)
    possibleworldsnet = pwn.PossibleWorlds(nn, args.n_worlds, args.d_world)

    opt = tf.train.AdamOptimizer()

    losses = []
    accuracys = []
    grad_norms = []
    for e in range(args.epochs):
        for A, B, E in data.fetch_data(args.batch_size):
            loss, grads, p = compute_step(possibleworldsnet, A, B, E)
            gnvs = zip(grads, possibleworldsnet.variables)
            step = tf.train.get_or_create_global_step()
            opt.apply_gradients(gnvs, global_step=step)

            acc =  np.mean(np.equal(np.round(p), np.array(E)))
            print('\rstep: {} loss {:.4f} acc {:.4f}'.format(step.numpy(),
                                tf.reduce_mean(loss), acc), end='', flush=True)
            losses.append(loss)
            accuracys.append(acc)
            # grad_norms.append(np.sum([np.linalg.norm(g) for g in grads]))

            if step.numpy() % 10 == 0:
                fig = plt.figure()
                plt.plot(losses, label='train loss')
                plt.plot(accuracys, label='train accuracy')
                # plt.plot(grad_norms, label='train grad norms')
                plt.title('PWN at step {}'.format(str(step.numpy())))
                plt.legend()
                plt.savefig('/tmp/train_pwn.png')
                plt.close()


if __name__ == "__main__":
    tf.enable_eager_execution()
    main(argumentparser())
