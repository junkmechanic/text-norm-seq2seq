from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell


class TransNormModel(object):
    """
    This class represents the Encoder-Decoder model used for normalization.
    """

    def __init__(
        self,
        encoder_seq,
        decoder_seq,
        vocab_size,
        batch_size=128,
        cell_size=1024,
        num_layers=1,
        max_gradient_norm=5.0,
        learning_rate=0.0005,
        learning_rate_decay_factor=0.99,
        use_lstm=True,
        forward_only=False,
        prediction=False,
    ):
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

        self.encoder_seq = encoder_seq
        self.decoder_seq = decoder_seq
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.use_lstm = use_lstm
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.forward_only = forward_only
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                                         name="learning_rate")
        # this decay op can be replaced by an tf.train.exponential_decay, but
        # later
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # Building the graph

        self.build_embedding()

        # the embedded encoder input is a tensor of shape
        # [batch_size, max_seq_len, embedding_size]
        self.encoder_output, self.encoder_state = self.build_encoder()
        # encoder_output is a tensor of shape
        # [batch_size, max_seq_len, cell_size]
        # the encoder_output will be used for attention
        # encoder_state is a tuple, representing the final RNN state, of tensors
        # of shapes `[batch_size, s] for s in cell.state_size`
        # the encoder_state will be used as the initial state for the decoder

        # the embedded decoder input is a tensor of shape
        # [batch_size, max_seq_len, embedding_size]
        # if not prediction:
        self.decoder_output, _ = self.build_decoder()

    def build_embedding(self):
        with vs.variable_scope("embedding"):
            self.embedding = _get_variable(name="char_embedding",
                                           shape=[self.vocab_size,
                                                  self.cell_size], wd=0.0)
            self.embedded_encoder_input = embedding_ops.embedding_lookup(
                self.embedding, self.encoder_seq)
            # decoder inputs are embedded using the same embedding
            self.embedded_decoder_input = embedding_ops.embedding_lookup(
                self.embedding, self.decoder_seq)

    def build_encoder(self):
        with vs.variable_scope("encoder_layer") as scope:
            if self.use_lstm:
                single_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size,
                                                           state_is_tuple=True)
            else:
                single_cell = tf.nn.rnn_cell.GRUCell(self.cell_size)
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] *
                                                   self.num_layers)
            else:
                cell = single_cell
            lengths = get_sequence_lengths(self.embedded_encoder_input, scope)
            outputs, final_state = rnn.dynamic_rnn(cell,
                                                   self.embedded_encoder_input,
                                                   dtype=tf.float32,
                                                   sequence_length=lengths,
                                                   time_major=False)

            # final_state is a tuple LSTMStateTuple(state c, output h)
            return outputs, final_state

    def build_decoder(self):
        with vs.variable_scope("decoder_layer") as scope:
            if self.use_lstm:
                single_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size,
                                                           state_is_tuple=True)
            else:
                single_cell = tf.nn.rnn_cell.GRUCell(self.cell_size)
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] *
                                                   self.num_layers)
            else:
                cell = single_cell
            # The output from the decoder RNN needs to pass through fully
            # connected feed forward layer for sequence prediction
            cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, self.vocab_size)

            if not self.forward_only:
                # this part will build the graph for learning
                # the decoder outputs will be generated by dynamic unrolling of
                # the RNN over the decoder input sequence
                lengths = get_sequence_lengths(self.embedded_decoder_input,
                                               scope)
                outputs, final_state = rnn.dynamic_rnn(
                    cell, self.embedded_decoder_input, sequence_length=lengths,
                    initial_state=self.encoder_state, dtype=tf.float32,
                    time_major=False)
                return outputs, final_state[0]
            else:
                return self.manual_unroll(cell)

    def manual_unroll(self, cell):
        with vs.variable_scope("rnn_unroll"):
            # this part will build the graph for testing/validation
            # the decoder will use the first symbol, _GO, as the first input
            # and will feed the output of the current step as the input for
            # the next step.

            transposed = tf.transpose(self.embedded_decoder_input, [1, 0, 2],
                                      name='transpose_decoder_input')
            # transposed is a tensor of shape
            # [max_seq_len, batch_size, cell_size]
            time_steps = tf.shape(transposed)[0]
            int_time_steps = transposed.get_shape().with_rank(3).as_list()[0]
            # the transposed tensor will be unpacked so that each batch time
            # step can be extracted for decoding
            # Since max_seq_len is not know, tf.unpack cannot be used.
            # TensorArray is to be used.
            input_ta = tf.TensorArray(
                dtype=transposed.dtype,
                size=tf.shape(transposed)[0],
                tensor_array_name='dynamic_unpacked_input'
            )
            input_ta = input_ta.unpack(transposed)
            output_ta = tf.TensorArray(
                dtype=transposed.dtype,
                size=tf.shape(transposed)[0],
                tensor_array_name='dynamic_output'
            )
            time = tf.constant(0, dtype=tf.int32, name="time")
            state = self.encoder_state
            state = rnn_cell._unpacked_state(state)
            prev = input_ta.read(time)

            def _time_step(time, prev, output_ta_t, *state):
                state = (rnn_cell._packed_state(structure=cell.state_size,
                                                state=state))
                output, new_state = cell(prev, state)
                output_ta_t.write(time, output)
                prev_symbol = tf.argmax(output, 1)
                # prev_symbol is of shape [batch_size, 1] holding the index of
                # the predicted character in the vocab. This needs to be
                # embedded to be used as the input to the decoder for the next
                # time step.
                emb_prev = embedding_ops.embedding_lookup(
                    self.embedding, prev_symbol
                )
                new_state = tuple(rnn_cell._unpacked_state(new_state))
                return (time + 1, emb_prev, output_ta_t) + new_state

            unrolled_vars = tf.while_loop(
                cond=lambda time, *_: time < time_steps,
                body=_time_step,
                loop_vars=(time, prev, output_ta) + tuple(state),
            )
            output_ta_final, state_final = unrolled_vars[2], unrolled_vars[3]

            output_final = output_ta_final.pack()
            output_final.set_shape([int_time_steps, self.batch_size,
                                    cell.output_size])
            output_final = tf.transpose(output_final, [1, 0, 2],
                                        name='transpose_decoder_output')
            return output_final, state_final


def _get_variable(name, shape, wd=None):
    var = tf.get_variable(name, shape)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def get_sequence_lengths(sequence_batch, scope):
    with vs.variable_scope("gen_seq_len"):
        lengths = tf.reduce_sum(
            tf.sign(tf.reduce_max(sequence_batch, reduction_indices=2)),
            reduction_indices=1,
            name="sequence_lengths"
        )
        return lengths
