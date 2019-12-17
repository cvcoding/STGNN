#############################################################################################

### 4.3 Language modeling ###
    
### See at the beginning of the file !!! ###

import collections
import os
import sys

import numpy as np
import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.char.train.txt")
  valid_path = os.path.join(data_path, "ptb.char.valid.txt")
  test_path = os.path.join(data_path, "ptb.char.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  
  return np.asarray(train_data), np.asarray(valid_data), np.asarray(test_data), vocabulary


def ptb_producer(raw_data, num_steps, num_classes):
  """Iterate on the raw PTB data.
  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    num_classes: int, the number of classes
  Returns:
    A pair of Tensors, each shaped [*, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
  """

  data_len = len(raw_data)
  batch_len = (data_len // num_steps) * num_steps
  
  X = np.expand_dims(raw_data[:batch_len].reshape((-1, num_steps)), axis=-1)
  y = np.expand_dims(raw_data[1:(batch_len+1)].reshape((-1, num_steps)), axis=-1)
  y = np.squeeze(np.eye(num_classes)[y])

  return X, y