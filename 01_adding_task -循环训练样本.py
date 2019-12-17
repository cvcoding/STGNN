"""
Train RNN models on the adding task. The network is given a sequence of (value, marker) tuples. The desired output is
the addition of the only two values that were marked with a 1, whereas those marked with a 0 need to be ignored.
Markers appear only in the first 10% and last 50% of the sequences.

Validation is performed on data generated on the fly.
"""

from __future__ import absolute_import
from __future__ import print_function

import random
import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers

from util.misc import *
from util.graph_definition import *
from util.prepare_data import load_data
from util.ops import (get_minibatches_indices, pad_seqs,
                          print_metrics,
                          get_max_length, calculate_metrics, create_conf_dict,
                          save_config_dict, get_validset_feeds)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # noqa

# Task-independent flags
create_generic_flags()
sequence_length_num = 120
# Task-specific flags
tf.app.flags.DEFINE_integer('validation_batches', 50, "How many batches to use for validation metrics.")
tf.app.flags.DEFINE_integer('evaluate_every', 300, "How often is the model evaluated.")
tf.app.flags.DEFINE_integer('sequence_length', sequence_length_num, "Sequence length.")

FLAGS = tf.app.flags.FLAGS

# Constants
MIN_VAL = -0.5
MAX_VAL = 0.5
FIRST_MARKER = 10.
SECOND_MARKER = 10.
INPUT_SIZE = 2
OUTPUT_SIZE = 1
EPECHS = 100
weight_decay_rate = 0.0


def task_setup():
    print('\tSequence length: %d' % FLAGS.sequence_length)
    print('\tValues drawn from Uniform[%.1f, %.1f]' % (MIN_VAL, MAX_VAL))
    print('\tFirst marker: first %d%%' % FIRST_MARKER)
    print('\tSecond marker: last %d%%' % SECOND_MARKER)


def generate_example(seq_length, min_val, max_val):
    """
    Creates a list of (a,b) tuples where a is random[min_val,max_val] and b is 1 in only
    two tuples, 0 for the rest. The ground truth is the addition of a values for tuples with b=1.

    :param seq_length: length of the sequence to be generated
    :param min_val: minimum value for a
    :param max_val: maximum value for a

    :return x: list of (a,b) tuples
    :return y: ground truth
    """
    # Select b values: one in first X% of the sequence, the other in the second Y%
    b1 = random.randint(0, int(seq_length * FIRST_MARKER / 100.) - 1)
    b2 = random.randint(0, int(seq_length * FIRST_MARKER / 100.) - 1) + seq_length*4//5
    # b2 = random.randint(int(seq_length * SECOND_MARKER / 100.), seq_length - 1)

    b = [0.] * seq_length
    b[b1] = 1.
    b[b2] = 1.

    # Generate list of tuples
    x = [(random.uniform(min_val, max_val), marker) for marker in b]
    y = x[b1][0] + x[b2][0]

    return x, y


def generate_batch(seq_length, batch_size, min_val, max_val):
    """
    Generates batch of examples.

    :param seq_length: length of the sequence to be generated
    :param batch_size: number of samples in the batch
    :param min_val: minimum value for a
    :param max_val: maximum value for a

    :return x: batch of examples
    :return y: batch of ground truth values
    """
    n_elems = 2
    x = np.empty((batch_size, seq_length, n_elems))
    y = np.empty((batch_size, 1))

    for i in range(batch_size):
        sample, ground_truth = generate_example(seq_length, min_val, max_val)
        x[i, :, :] = sample
        y[i, 0] = ground_truth
    return x, y


def train():
    data_list = load_data('add', seq_len=sequence_length_num)
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_list

    samples = tf.placeholder(tf.float32, [None, None, INPUT_SIZE])  # (batch, time, in)
    ground_truth = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])  # (batch, out)

    cell, initial_state = create_model(model=FLAGS.model,
                                       num_cells=[FLAGS.rnn_cells1, FLAGS.rnn_cells2],
                                       # num_cells=[FLAGS.rnn_cells] * FLAGS.rnn_layers,
                                       batch_size=FLAGS.batch_size)

    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, samples, dtype=tf.float32, initial_state=initial_state)

    # Split the outputs of the RNN into the actual outputs and the state update gate
    rnn_outputs, updated_states = split_rnn_outputs(FLAGS.model, rnn_outputs)

    out = layers.linear(inputs=rnn_outputs[:, -1, :], num_outputs=OUTPUT_SIZE)

    # Compute L2 loss
    # mse = tf.nn.l2_loss(ground_truth - out) / FLAGS.batch_size
    mean_squared_error = tf.losses.mean_squared_error(
        labels=ground_truth, predictions=out)
    mse1 = tf.reduce_mean(mean_squared_error)
    tf.summary.scalar('mean_squared_error',
                      tf.reduce_mean(mean_squared_error))
    weight_decay = weight_decay_rate * tf.add_n([tf.nn.l2_loss(v) for v in
                                                 tf.trainable_variables()
                                                 if 'bias' not in v.name])
    tf.summary.scalar('weight_decay', weight_decay)
    mse = mse1 + weight_decay

    ## Compute loss for each updated state
    budget_loss = compute_budget_loss(FLAGS.model, mse, updated_states, FLAGS.cost_per_sample)
    loss = mse + budget_loss
    # loss = mse

    # Optimizer
    opt, grads_and_vars = compute_gradients(loss, FLAGS.learning_rate, FLAGS.grad_clip)
    train_fn = opt.apply_gradients(grads_and_vars)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    for e in range(EPECHS):
        minibatch_indices = get_minibatches_indices(
            len(x_train), FLAGS.batch_size)
        # eval_metrics = np.zeros(len(metric_tags))

        for b_num, b_indices in enumerate(minibatch_indices):
            # print('\rProcessing batch {}/{}'.format(
            #     b_num, len(minibatch_indices)), end='', flush=True)
            x = [x_train[i] for i in b_indices]
            y = y_train[b_indices]
            x, seq_lengths = pad_seqs(x)
            # # Generate new batch and perform SGD update
            # x, y = generate_batch(min_val=MIN_VAL, max_val=MAX_VAL,
            #                       seq_length=FLAGS.sequence_length,
            #                       batch_size=FLAGS.batch_size)

            sess.run([train_fn], feed_dict={samples: x, ground_truth: y})

        # Evaluate on validation data generated on the fly
        valid_error, valid_steps = 0., 0.
        for _ in range(FLAGS.validation_batches):
            valid_x, valid_y = generate_batch(min_val=MIN_VAL, max_val=MAX_VAL,
                                                      seq_length=FLAGS.sequence_length,
                                                      batch_size=FLAGS.batch_size)
            valid_iter_error, valid_used_inputs = sess.run(
                [mse, updated_states],
                feed_dict={
                    samples: valid_x,
                    ground_truth: valid_y})
            valid_error += valid_iter_error
            if valid_used_inputs is not None:
                valid_steps += compute_used_samples(valid_used_inputs)
            else:
                valid_steps += FLAGS.sequence_length
        valid_error /= FLAGS.validation_batches
        valid_steps /= FLAGS.validation_batches
        print("epech: %d,"
              "validation error: %.7f, "
                "validation samples: %.2f%%" % (e, valid_error,
                                                  100. * valid_steps / FLAGS.sequence_length))


def main(argv=None):
    print_setup(task_setup)
    train()


if __name__ == '__main__':
    tf.app.run()
