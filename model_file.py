import tensorflow as tf
from aux_code.rnn_cells import CustomLSTMCell, ScdLSTMCell
from aux_code.tf_ops import linear
import rnn_local, rnn_local_scd
import rnn_cell_impl_local, rnn_cell_impl_local_scd
from tensorflow.python.framework import ops
import numpy as np


def rnn(x, h_dim, y_dim, keep_prob, sequence_lengths,
        training, output_format, cell_type='janet',
        t_max=None, batch_size=0):
    '''
    Inputs:
    x - The input data.
        Tensor shape (batch_size, max_sequence_length, n_features)
    h_dim - A list with the number of neurons in each hidden layer
    keep_prob - The percentage of weights to keep in each iteration
                 (1-drop_prob)

    Returns:
    The non-softmaxed output of the LSTM.
    '''

    def _binary_round(x):
        g = tf.get_default_graph()
        with ops.name_scope("BinaryRound") as name:
            with g.gradient_override_map({"Round": "Identity"}):
                return tf.round(x, name=name)

    def init_mask(t_max):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            bias_j = 0.5*tf.ones([t_max, 1])
            return bias_j
        return _initializer

    def init_u(t_max):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            bias_j = 1*tf.ones([1, t_max])
            return bias_j
        return _initializer

    def init_D(t_max):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            bias_j = 1*tf.ones([1, t_max])  # 0.5
            return bias_j
        return _initializer

    def weight_initializer(num_gates):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            p = np.zeros(shape)
            num_units = int(shape[0] // num_gates)
            p[-num_units:] = 1*np.ones(num_units)
            return tf.constant(p, dtype)
        return _initializer

    def bias_initializer(num_gates):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            p = np.zeros(shape)
            num_units = int(shape[0] // num_gates)
            p[-num_units:] = 1*np.zeros(num_units)
            return tf.constant(p, dtype)
        return _initializer

    def fst_cell(dim, output_projection=None):
        cell = CustomLSTMCell(dim, t_max=t_max, forget_only=True)
        drop_cell = rnn_cell_impl_local.DropoutWrapper(
            cell,
            input_keep_prob=1,
            output_keep_prob=keep_prob)
        return drop_cell

    def scd_cell(dim, output_projection=None):
        cell = ScdLSTMCell(dim, t_max=t_max, forget_only=True)
        drop_cell = rnn_cell_impl_local_scd.DropoutWrapper(
            cell,
            input_keep_prob=1,
            output_keep_prob=keep_prob)
        return drop_cell

    fstcell = fst_cell(h_dim[0])
    scdcell = scd_cell(h_dim[1])

    if output_format == 'last':

        x = tf.layers.batch_normalization(x)
        mask_weight = tf.get_variable('mask_weight', [1, 1], initializer=weight_initializer(1))

        # mask_weight = tf.get_variable('mask_weight', [1, 1])
        mask_bias = tf.get_variable('mask_bias', [1], initializer=bias_initializer(1))
        x1 = tf.reshape(x, [batch_size*t_max, 1])

        mask = tf.nn.bias_add(tf.matmul(x1, mask_weight), mask_bias)
        mask = tf.reshape(mask, [batch_size, t_max])

        u = tf.get_variable('u', [1, t_max], initializer=init_u(t_max))
        u = tf.tile(u, [batch_size, 1])
        D = tf.get_variable('D', [1, t_max], initializer=init_D(t_max))
        D = tf.tile(D, [batch_size, 1])
        D_diag = tf.matrix_diag(D)

        # sparse func
        mask = 1-tf.nn.relu(0.5*(tf.nn.tanh(mask + u) + tf.nn.tanh(mask - u)))
        # mask = tf.sigmoid(mask)
        # mask = 1 - _binary_round(mask)
        mask = _binary_round(mask)
        mask = tf.expand_dims(mask, 2)
        mask_weighted = tf.matmul(D_diag, mask)

        new_x = tf.concat([x, mask], 2)
        out, _ = rnn_local.dynamic_rnn(
            fstcell, new_x, sequence_length=sequence_lengths,
            dtype=tf.float32)

        # out = tf.layers.batch_normalization(out)

        out = out * mask_weighted
        new_out = tf.concat([out, mask], 2)
        mask1dim_out = tf.squeeze(mask)

        _, final_state = rnn_local_scd.dynamic_rnn(
            scdcell, new_out, sequence_length=sequence_lengths,
            dtype=tf.float32)

        out = final_state
        if cell_type == 'lstm' or cell_type == 'Reslstm':
            out = out[1]

        # out = tf.layers.batch_normalization(out)

        proj_out = linear(out, y_dim, scope='output_mapping')
    elif output_format == 'all':
        out, _ = tf.nn.dynamic_rnn(
            fstcell, x, sequence_length=sequence_lengths,
            dtype=tf.float32)
        flat_out = tf.reshape(out, (-1, out.get_shape()[-1]))
        proj_out = linear(flat_out, y_dim, scope='output_mapping')
        proj_out = tf.reshape(proj_out,
                              (tf.shape(out)[0], tf.shape(out)[1], y_dim))

    return proj_out, mask1dim_out


class RNN_Model(object):

    def __init__(self, n_features, n_classes, h_dim, max_sequence_length,
                 is_test=False, max_gradient_norm=None, opt_method='adam',
                 learning_rate=0.001, weight_decay=0,
                 cell_type='lstm', chrono=False, mse=False, batch_size=0,
                 ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.h_dim = h_dim
        self.max_sequence_length = max_sequence_length
        self.opt_method = opt_method
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.is_test = is_test
        self.weight_decay = weight_decay
        self.cell_type = cell_type
        self.chrono = chrono
        self.mse = mse
        self.batch_size = batch_size

    def build_inputs(self):
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.x = tf.placeholder(tf.float32, [None, None, self.n_features],
                                name='x')
        if self.output_seq:
            self.y = tf.placeholder(tf.float32, [None, None, self.n_classes],
                                    name='y')
        else:
            self.y = tf.placeholder(tf.float32, [None, self.n_classes],
                                    name='y')
        self.seq_lens = tf.placeholder(tf.int32, [None],
                                       name="sequence_lengths")
        self.training = tf.placeholder(tf.bool)

    def build_loss(self, outputs, mask):
        if self.mse:
            mean_squared_error = tf.losses.mean_squared_error(
                labels=self.y, predictions=outputs)
            self.loss_nowd = tf.reduce_mean(mean_squared_error)
            tf.summary.scalar('mean_squared_error',
                              tf.reduce_mean(mean_squared_error))
        else:
            if self.output_seq:
                flat_out = tf.reshape(outputs, (-1, tf.shape(outputs)[-1]))
                flat_y = tf.reshape(self.y, (-1, tf.shape(self.y)[-1]))
                sample_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=flat_y, logits=flat_out)

            else:
                sample_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y, logits=outputs)

            tf.summary.scalar('cross_entropy',
                              tf.reduce_mean(sample_cross_entropy))
            self.loss_nowd = tf.reduce_mean(sample_cross_entropy)

        weight_decay = self.weight_decay*tf.add_n([tf.nn.l2_loss(v) for v in
                                                   tf.trainable_variables()
                                                   if 'bias' not in v.name])
        tf.summary.scalar('weight_decay', weight_decay)

        mask_norm = 1e-6*tf.norm(mask, 1)  #tf.nn.l2_loss(mask)
        self.loss = self.loss_nowd + weight_decay + mask_norm

    def build_optimizer(self):
        if self.opt_method == 'adam':
            print('Optimizing with Adam')
            opt = tf.train.AdamOptimizer(self.learning_rate)
        elif self.opt_method == 'rms':
            print('Optimizing with RMSProp')
            opt = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.opt_method == 'momentum':
            print('Optimizing with Nesterov momentum SGD')
            opt = tf.train.MomentumOptimizer(self.learning_rate,
                                             momentum=0.9,
                                             use_nesterov=True)

        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         self.max_gradient_norm)
        tf.summary.scalar('gradients_norm', norm)

        self.train_opt = opt.apply_gradients(zip(clipped_gradients, params))

    def build(self, output_format='last'):
        print("Building model ...")
        self.output_seq = False
        if output_format == 'all':
            self.output_seq = True
        self.build_inputs()

        t_max = None
        if self.chrono:
            t_max = self.max_sequence_length

        outputs, mask = rnn(self.x, self.h_dim, self.n_classes, self.keep_prob,
                      sequence_lengths=self.seq_lens,
                      training=self.training, output_format=output_format,
                      cell_type=self.cell_type, t_max=t_max, batch_size=self.batch_size,
                      )
        self.mask = mask

        self.build_loss(outputs, mask)

        self.output_probs = tf.nn.softmax(outputs)
        if self.output_seq:
            self.output_probs = tf.reshape(
                self.output_probs, (-1, tf.shape(self.output_probs)[-1]))

        if not self.is_test:
            print("Adding training operations")
            self.build_optimizer()

        self.summary_op = tf.summary.merge_all()
