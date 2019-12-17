"""
Train RNN models on sequential MNIST, where inputs are processed pixel by pixel.

Results should be reported by evaluating on the test set the model with the best performance on the validation set.
To avoid storing checkpoints and having a separate evaluation script, this script evaluates on both validation and
test set after every epoch.
"""

from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers

from tensorflow.examples.tutorials.mnist import input_data

from util.misc import *
from util.graph_definition import *
import numpy as np
import random
import copy
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # noqa
# Task-independent flags
create_generic_flags()

# Task-specific flags
tf.app.flags.DEFINE_string('data_path', './MNIST', 'Path where the MNIST data will be stored.')

FLAGS = tf.app.flags.FLAGS

# Constants
OUTPUT_SIZE = 10
SEQUENCE_LENGTH = 784
VALIDATION_SAMPLES = 5000
NUM_EPOCHS = 100

# Load data
mnist = input_data.read_data_sets(FLAGS.data_path, one_hot=False, validation_size=VALIDATION_SAMPLES)
ITERATIONS_PER_EPOCH = int(mnist.train.num_examples / FLAGS.batch_size)
VAL_ITERS = int(mnist.validation.num_examples / FLAGS.batch_size)
TEST_ITERS = int(mnist.test.num_examples / FLAGS.batch_size)


def pad_seqs(seqs):
    """Create the matrices from the datasets.

    This pad each sequence to the same length: the length of the
    longest sequence.

    Output:
    x -- a numpy array with shape (batch_size, max_time_steps, num_features)
    lengths -- an array of the sequence lengths

    """
    lengths = [len(s) for s in seqs]

    n_samples = len(seqs)
    maxlen = np.max(lengths)
    inputDimSize = seqs[0].shape[1]

    # x = np.zeros((n_samples, maxlen, inputDimSize))
    x = np.zeros((n_samples, maxlen))

    for idx, s in enumerate(seqs):
        x[idx, :lengths[idx]] = np.reshape(s, [784])
    return x, np.array(lengths)


def get_minibatches_indices(n, minibatch_size, shuffle=True,
                            shuffleBatches=True,
                            allow_smaller_final_batch=False):
    """
    Inputs:
    n - An integer corresponding to the total number of data points

    Returns:
    minibatches - a list of lists with indices in the lowest dimension
    """
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    if n <= minibatch_size:
        return [idx_list]

    minibatches = []
    minibatch_start = 0

    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size
    if allow_smaller_final_batch:
        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

    if shuffleBatches is True:
        # minibatches here is a list of list indexes
        random.shuffle(minibatches)

    return minibatches


def train():
    samples_raw = tf.placeholder(tf.float32, [None, None])  # (batch, 28, 28)
    samples = tf.expand_dims(samples_raw, -1)  # (batch, 28*28, 1)
    ground_truth = tf.placeholder(tf.int64, [None])  # (batch)
    # two layers of skip rnn : num_cells=[FLAGS.rnn_cells1, FLAGS.rnn_cells2],  # * FLAGS.rnn_layers
    cell, initial_state = create_model(model=FLAGS.model,
                                       num_cells=[FLAGS.rnn_cells1],  #, FLAGS.rnn_cells2 * FLAGS.rnn_layers
                                       batch_size=FLAGS.batch_size)

    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, samples, dtype=tf.float32, initial_state=initial_state)

    # Split the outputs of the RNN into the actual outputs and the state update gate
    rnn_outputs, updated_states = split_rnn_outputs(FLAGS.model, rnn_outputs)

    out = layers.linear(inputs=rnn_outputs[:, -1, :], num_outputs=OUTPUT_SIZE)

    # Compute cross-entropy loss
    cross_entropy_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=ground_truth)
    cross_entropy = tf.reduce_mean(cross_entropy_per_sample)

    # Compute accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), ground_truth), tf.float32))

    # Compute loss for each updated state
    budget_loss = compute_budget_loss(FLAGS.model, cross_entropy, updated_states, FLAGS.cost_per_sample)

    # Combine all losses
    loss = cross_entropy + budget_loss

    # Optimizer
    opt, grads_and_vars = compute_gradients(loss, FLAGS.learning_rate, FLAGS.grad_clip)
    train_fn = opt.apply_gradients(grads_and_vars)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    perm_mask = np.load('misc/pmnist_permutation_mask.npy')
    x_train = list(np.expand_dims(mnist.train.images[:,perm_mask],-1))
    y_train = mnist.train.labels
    x_valid = list(np.expand_dims(mnist.validation.images[:,perm_mask],-1))
    y_valid = mnist.validation.labels
    x_test = list(np.expand_dims(mnist.test.images[:,perm_mask], -1))
    y_test = mnist.test.labels

    try:
        # number of trainable params
        t_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print('Number of trainable params=', t_params)

        for epoch in range(NUM_EPOCHS):
            minibatch_indices = get_minibatches_indices(
                len(x_train), FLAGS.batch_size)
            for b_num, b_indices in enumerate(minibatch_indices):
                # Generate new batch and perform SGD update
                x = [x_train[i] for i in b_indices]
                y = y_train[b_indices]
                x, seq_lengths = pad_seqs(x)

                # x, y = mnist.train.next_batch(FLAGS.batch_size)
                sess.run([train_fn], feed_dict={samples_raw: x, ground_truth: y})

            # Evaluate on validation data
            # valid_accuracy, valid_steps = 0, 0
            # for _ in range(VAL_ITERS):
            #     valid_x, valid_y = mnist.validation.next_batch(FLAGS.batch_size)
            #     valid_iter_accuracy, valid_used_inputs = sess.run(
            #         [accuracy, updated_states],
            #         feed_dict={
            #             samples_raw: valid_x,
            #             ground_truth: valid_y})
            #     valid_accuracy += valid_iter_accuracy
            #     if valid_used_inputs is not None:
            #         valid_steps += compute_used_samples(valid_used_inputs)
            #     else:
            #         valid_steps += SEQUENCE_LENGTH
            # valid_accuracy /= VAL_ITERS
            # valid_steps /= VAL_ITERS

            # Evaluate on test data

            test_accuracy, test_steps = 0, 0
            minibatch_indices_t = get_minibatches_indices(
                len(x_test), FLAGS.batch_size)
            for b_num_t, b_indices_t in enumerate(minibatch_indices_t):
                # Generate new batch and perform SGD update
                test_x = [x_test[i] for i in b_indices_t]
                test_y = y_test[b_indices_t]
                test_x, seq_lengths = pad_seqs(test_x)
            # for _ in range(TEST_ITERS):
            #     test_x, test_y = mnist.test.next_batch(FLAGS.batch_size)
                test_iter_accuracy, test_used_inputs = sess.run(
                    [accuracy, updated_states],
                    feed_dict={
                        samples_raw: test_x,
                        ground_truth: test_y})
                test_accuracy += test_iter_accuracy
                if test_used_inputs is not None:
                    test_steps += compute_used_samples(test_used_inputs)
                else:
                    test_steps += SEQUENCE_LENGTH
            test_accuracy /= TEST_ITERS
            test_steps /= TEST_ITERS

            print("Epoch %d/%d, "
                  "test accuracy: %.2f%%, " % (epoch + 1,
                                                   NUM_EPOCHS,
                                                   100. * test_accuracy,))
    except KeyboardInterrupt:
        pass


def main(argv=None):
    print_setup()
    train()


if __name__ == '__main__':
    tf.app.run()
