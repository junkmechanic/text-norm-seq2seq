from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn


class TransNormModel(object):
    """
    This class represents the Encoder-Decoder model used for normalization.
    """

    def __init__(self,
                 encoder_seq,
                 decoder_seq,
                 vocab_size,
                 batch_size,
                 cell_size,
                 num_layers,
                 max_gradient_norm,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=True,
                 forward_only=False):
        """
        The class will not hold variables from the graph, whereever possible.

        Args:
            encoder_seq : a minibatch of encoder sequences in batch-major format
                          with shape [batch_size, max_seq_len].
                          The max_seq_len can vary.
            decoder_seq : a minibatch of decoder sequences in batch-major format
                          with shape [batch_size, max_seq_len]
        """
        # TODO:
        # define arguments
        # experiments:
        #  - add attention mechanism
        #  - change weight decay
        #  - add bidirectional rnn
        #  - add dropout
        # See if adding cpu as the device to create the variable on makes a
        # difference

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.use_lstm = use_lstm
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                                         name="learning_rate")
        # this decay op can be replaced by an tf.train.exponential_decay, but
        # later
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # Building the graph

        with vs.variable_scope("embedding") as scope:
            self.embedded_encoder_input = embed(encoder_seq, self.vocab_size,
                                                cell_size, scope)
            scope.reuse_variables()
            # decoder inputs are embedded using the same embedding
            self.embedded_decoder_input = embed(decoder_seq, self.vocab_size,
                                                cell_size, scope)

        # the embedded encoder input is a tensor of shape
        # [batch_size, max_seq_len, embedding_size]
        self.encoder_output, self.encoder_state = build_encoder(
            self.embedded_encoder_input, self.cell_size, self.num_layers,
            self.use_lstm
        )
        # encoder_output is a tensor of shape
        # [batch_size, max_seq_len, cell_size]
        # the encoder_output will be used for attention
        # encoder_state is a tensor, representing the final RNN state, of shape
        # [batch_size, cell_size]
        # the encoder_state will be used as the initial state for the decoder

        # the embedded decoder input is a tensor of shape
        # [batch_size, max_seq_len, embedding_size]
        self.decoder_output, _ = build_decoder(self.embedded_decoder_input,
                                               self.encoder_state,
                                               self.encoder_output,
                                               self.cell_size, self.num_layers,
                                               self.vocab_size, use_lstm,
                                               forward_only)


def _get_variable(name, shape, wd=None):
    var = tf.get_variable(name, shape)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def embed(to_embed, domain_size, embedding_size, scope):
    with vs.variable_scope(scope or "embedding"):
        embedding = _get_variable(name="char_embedding",
                                  shape=[domain_size, embedding_size],
                                  wd=0.0)
        embedded = embedding_ops.embedding_lookup(embedding, to_embed)
        return embedded


def get_sequence_lengths(sequence_batch, scope):
    with vs.variable_scope(scope):
        lengths = tf.reduce_sum(
            tf.sign(tf.reduce_max(sequence_batch, reduction_indices=2)),
            reduction_indices=1,
            name="sequence_lengths"
        )
        return lengths


def build_encoder(input_batch, cell_size, num_layers, use_lstm):
    with vs.variable_scope("encoder_layer") as scope:
        if use_lstm:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size,
                                                       state_is_tuple=True)
        else:
            single_cell = tf.nn.rnn_cell.GRUCell(cell_size)
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
        else:
            cell = single_cell
        lengths = get_sequence_lengths(input_batch, scope)
        outputs, final_state = rnn.dynamic_rnn(cell, input_batch,
                                               dtype=tf.float32,
                                               sequence_length=lengths,
                                               time_major=False)

        # final_state is a tuple LSTMStateTuple(state c, output h)
        return outputs, final_state[0]


def build_decoder(input_batch, intital_state, attention_state, cell_size,
                  num_layers, output_size, use_lstm, forward_only):
    with vs.variable_scope("decoder_layer") as scope:
        if use_lstm:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size,
                                                       state_is_tuple=True)
        else:
            single_cell = tf.nn.rnn_cell.GRUCell(cell_size)
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
        else:
            cell = single_cell
        # The output from the decoder RNN needs to pass through fully connected
        # feed forward layer for sequence prediction
        cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, output_size)

        # if forward_only is true (generally during prediction), the output of
        # the previous step needs is given as the input for the current step
        # TODO: feed previous

        lengths = get_sequence_lengths(input_batch, scope)
        outputs, final_state = rnn.dynamic_rnn(cell, input_batch,
                                               dtype=tf.float32,
                                               sequence_length=lengths,
                                               time_major=False)
        return outputs, final_state[0]
