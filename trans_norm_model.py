from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time
from datetime import datetime

import numpy as np
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
        targets,
        vocab_size,
        batch_size=128,
        cell_size=1024,
        num_layers=1,
        max_gradient_norm=5.0,
        learning_rate=0.0005,
        learning_rate_decay_factor=0.99,
        use_lstm=True,
        forward_only=False,
        predict=False,
        model_dir='./model',
        eos_idx=8,
        max_pred_length=50,
    ):
        """
        The class will not hold variables from the graph, whereever possible.

        Args:
            encoder_seq : minibatch of encoder sequences in batch-major format
                with shape [batch_size, max_seq_len].
                The max_seq_len can vary.
            decoder_seq : minibatch of decoder sequences in batch-major format
                with shape [batch_size, max_seq_len]
            forward_only : boolean flag to indicate if the backpropagation part
                of the model is to be built or not. This is generally set when
                the model graph is being rebuilt for validation and test sets.
            predict : boolean to indicate if the model is being built for
                training or prediction. In case of prediction, decoder sequence
                is not expected and loss will not be calculated. decoder_seq
                should still be provided which would ideally just be the symbol
                _GO.

        A note regarding `forward_only` and `predict`. Either both would be true
        or both would be false or (forward_only=True and predict=False)
        """
        # TODO:
        # define arguments
        # add moving avg
        # experiments:
        #  - add attention mechanism
        #  - change weight decay
        #  - add bidirectional rnn
        #  - add dropout
        # See if adding cpu as the device to create the variable on makes a
        # difference

        self.encoder_seq = encoder_seq
        self.decoder_seq = decoder_seq
        self.targets = targets
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.use_lstm = use_lstm
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.max_gradient_norm = max_gradient_norm
        self.eos_idx = eos_idx
        self.max_pred_length = max_pred_length
        if not forward_only and predict:
            raise ValueError('!forward_only and predict are mutually exclusive')
        if not forward_only:
            self.run_level = 1
        else:
            self.run_level = 2
        if predict:
            self.run_level = 3

        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Building the graph
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                                         name="learning_rate")
        # this decay op can be replaced by an tf.train.exponential_decay, but
        # later
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

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
        self.decoder_output, _ = self.build_decoder()
        # the decoder_output tensor is of shape
        # [batch_size, max_seq_len, vocab_size]
        # tf.nn.sparse_softmax_cross_entropy_with_logits expects the input
        # `logits` to have shape [batch_size, num_classes]
        # for that the tensor decoder_output would have to be transposed and
        # then sliced using tf.slice.
        # It is easier to calculate the softmax entropy manually as we can then
        # apply the mask representing the individual length of sequences in the
        # batch
        self.predictions = self.make_predictions()
        # predictions has the same shape as decoder_output
        if self.run_level < 3:
            self.loss = self.compute_batch_loss()
        else:
            self.predicted_word = tf.nn.top_k(self.predictions, k=1)

        # update_model is a op that applies gradients to the variables (model
        # parameters)
        # this should be evaluated/run for the updates to happen, typically
        # during training
        if self.run_level < 2:
            self.update_model = self.backpropagate()

        # saving the model
        self.saver = tf.train.Saver(tf.all_variables())

        self.sess_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )

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

            if self.run_level == 1:
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
        """
        This method is written to mimic `tf.rnn.dynamic_rnn()` behavior. In fact
        most of it is a shameless copy of the code from
        `tf.rnn._dynamic_rnn_loop()`.
        Except for the fact that the input is not known to the decoder. Hence
        the output of the decoder at each time step will be utilized as its
        input for the next time step.
        This method also builds the graph for final prediction in which case the
        output is generated from scratch. Only the first symbol (`_GO`) is fed
        to the decoder.
        """

        # the name of the variable scope is to match that of the scope set by
        # tf.rnn.dynamic_rnn so that the variables can be restored
        with vs.variable_scope('RNN'):
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
            # Since max_seq_len is not known, tf.unpack cannot be used.
            # TensorArray is to be used.
            input_ta = tf.TensorArray(
                dtype=transposed.dtype,
                size=tf.shape(transposed)[0],
                tensor_array_name='dynamic_unpacked_input'
            )
            input_ta = input_ta.unpack(transposed)
            if self.run_level < 3:
                output_ta = tf.TensorArray(
                    dtype=transposed.dtype,
                    size=time_steps,
                    tensor_array_name='dynamic_output'
                )
            else:
                output_ta = tf.TensorArray(
                    dtype=transposed.dtype,
                    size=1,
                    dynamic_size=True,
                    tensor_array_name='dynamic_output'
                )
            time = tf.constant(0, dtype=tf.int32, name="time")
            state = self.encoder_state
            state = rnn_cell._unpacked_state(state)
            prev = input_ta.read(time)

            # the prediction round needs prev_symbol to decide when to stop the
            # RNN
            # this is a dummy value for the first iteration of the while loop.
            # it can be anything but the index for _EOS which is meant to stop
            # the while loop.
            dummy_val = [i for i in range(self.vocab_size)
                         if i != self.eos_idx][0]
            # the shape of the variable representing the previous symbol should
            # be [1] since the batch_size should ideally be 1 for prediction.
            prev_symbol_dummy = tf.constant([dummy_val], dtype=tf.int64)
            eos_val = tf.constant([self.eos_idx], dtype=tf.int64)
            max_len = tf.constant([self.max_pred_length], dtype=tf.int32)

            def pred_cond(time, prev, prev_symbol, output_ta_t, *state):
                return tf.reshape(
                    tf.logical_and(
                        tf.less(time, max_len),
                        tf.not_equal(prev_symbol, eos_val)
                    ), []
                )

            def _time_step(time, prev, prev_symbol, output_ta_t, *state):
                state = (rnn_cell._packed_state(structure=cell.state_size,
                                                state=state))
                # the reason for calling cell() here (as opposed to
                # tf.rnn._rnn_step() with sequence_lengths) is that
                # `manual_unroll` will only be called during the test or
                # prediction run. And in either case, it is to be assumed that
                # the output is not known.
                # During test, the max_seq_len of the batch will be used to run
                # the decoder for that many time steps.
                # During prediction, the decoder needs to check for the presence
                # of the `_EOS`.
                output, new_state = cell(prev, state)
                output_ta_t = output_ta_t.write(time, output)
                prev_symbol = tf.argmax(output, 1)
                # prev_symbol is of shape [batch_size, 1] holding the index of
                # the predicted character in the vocab. This needs to be
                # embedded to be used as the input to the decoder for the next
                # time step.
                emb_prev = embedding_ops.embedding_lookup(
                    self.embedding, prev_symbol
                )
                new_state = tuple(rnn_cell._unpacked_state(new_state))
                return (time + 1, emb_prev, prev_symbol, output_ta_t) + \
                    new_state

            if self.run_level < 3:
                unrolled_vars = tf.while_loop(
                    cond=lambda time, *_: time < time_steps,
                    body=_time_step,
                    loop_vars=(time, prev, prev_symbol_dummy, output_ta) +
                    tuple(state),
                )
            else:
                unrolled_vars = tf.while_loop(
                    cond=pred_cond,
                    body=_time_step,
                    loop_vars=(time, prev, prev_symbol_dummy, output_ta) +
                    tuple(state),
                )
            output_ta_final, state_final = unrolled_vars[3], unrolled_vars[4]

            output_final = output_ta_final.pack()
            output_final.set_shape([int_time_steps, self.batch_size,
                                    cell.output_size])
            output_final = tf.transpose(output_final, [1, 0, 2],
                                        name='transpose_decoder_output')
            return output_final, state_final

    def make_predictions(self):
        with tf.op_scope([self.decoder_output], name="prediction_layer"):
            # tf.nn.softmax takes inputs of rank 2, preferably of shape
            # [batch_size, num_classes]
            # decoder_output needs to be reshaped for this step
            decoder_output_shape = tf.shape(self.decoder_output)
            all_outputs = tf.reshape(self.decoder_output, [-1, self.vocab_size])
            predictions = tf.nn.softmax(all_outputs, name="softmax_output")
            predictions = tf.reshape(predictions, decoder_output_shape)
            predictions.set_shape(self.decoder_output.get_shape())
            return predictions

    def compute_batch_loss(self):
        with tf.op_scope([self.decoder_seq, self.predictions],
                         name="loss_computation"):
            # the targets are just decoder_inputs shifted by 1
            # shifted = tf.slice(self.decoder_seq, begin=[0, 1], size=[-1, -1])
            # zeros = tf.zeros([self.batch_size, 1], dtype=self.decoder_seq.dtype)
            # targets = tf.concat(1, [shifted, zeros])
            # the targets need to be in one-hot vector form.
            # tf.one_hot returns a tensor with type float32 by default
            one_hot_targets = tf.one_hot(self.targets, self.vocab_size)
            cross_entropy = one_hot_targets * tf.log(self.predictions)
            cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
            # cross_entropy is a tensor of shape [batch_size, max_seq_len]
            seq_indicators = tf.to_float(tf.sign(self.targets),
                                         name="cast_mask_float")
            # mask the cross_entropy
            cross_entropy *= seq_indicators

            # cost over each sequence
            cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
            cross_entropy /= tf.reduce_sum(seq_indicators, reduction_indices=1)

            # cost over the entire batch
            return tf.reduce_mean(cross_entropy)

    def backpropagate(self):
        with tf.op_scope([self.learning_rate, self.max_gradient_norm,
                          self.global_step], name="gradient_computation"):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads_vars = self.optimizer.compute_gradients(self.loss)
            grads, vars = [g for g, v in grads_vars], [v for g, v in grads_vars]
            clipped_grads, gnorm = tf.clip_by_global_norm(
                grads, self.max_gradient_norm
            )
            return self.optimizer.apply_gradients(zip(clipped_grads, vars),
                                                  global_step=self.global_step)

    def load_model(self, session, coordinator, model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print('{} : Reading model parameters from {}'.format(
                datetime.now().ctime(), ckpt.model_checkpoint_path))
            self.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print('{} : Creating model with fresh parameters'.format(
                datetime.now().ctime()))
            session.run(tf.initialize_all_variables())
        threads = tf.train.start_queue_runners(session, coordinator)
        return threads

    def learn(self, steps_per_checkpoint=200, model_dir=None):
        if self.run_level > 1:
            raise ValueError('Current run level doesnt support training.\n' +
                             'Check the flags `forward_only` and `predict`')
        if not model_dir:
            model_dir = self.model_dir
        ckpt_time, ckpt_loss = 0.0, 0.0
        loss_history = [float('inf')]
        with tf.Session(config=self.sess_conf) as sess:
            self.load_model(sess, None, model_dir)

            # training loop
            while True:
                start_time = time.time()
                _, step_loss = sess.run([self.update_model, self.loss])
                ckpt_time += (time.time() - start_time) / steps_per_checkpoint
                ckpt_loss += step_loss / steps_per_checkpoint

                step = self.global_step.eval()
                if step % steps_per_checkpoint == 0:
                    perplexity = math.expm1(ckpt_loss) if ckpt_loss < 300 else \
                        float('inf')
                    print('{} : step {} : learning rate {:8.7f} : '
                          'time {:3.2f} : perplexity {:10.9f}'.format(
                              datetime.now().ctime(), step,
                              self.learning_rate.eval(), ckpt_time, perplexity
                          ))

                    if ckpt_loss > max(loss_history[-3:]):
                        sess.run(self.learning_rate_decay_op)
                    loss_history.append(ckpt_loss)

                    # Save and reset
                    ckpt_path = os.path.join(model_dir, 'transNormModel.ckpt')
                    self.saver.save(sess, ckpt_path,
                                    global_step=self.global_step)
                    ckpt_time, ckpt_loss = 0.0, 0.0

    def test(self, num_examples, model_dir=None):
        """
        Ideally, the test would be done one example at a time.
        The batch_size should be set to 1
        """
        if self.run_level != 2:
            raise ValueError('Current run level doesnt support testing.\n' +
                             'Check the flags `forward_only` and `predict`')
        if not model_dir:
            model_dir = self.model_dir
        evaluations = []
        for seq_preds, seq_targets in zip(tf.unpack(self.predictions),
                                          tf.unpack(self.targets)):
            evaluations.append(tf.nn.in_top_k(seq_preds, seq_targets, 1))
        evals_int = tf.to_int32(evaluations)
        evals_int = tf.pack(evals_int)

        num_iter = num_examples / self.batch_size

        hits, misses = 0, 0
        with tf.Session(config=self.sess_conf) as sess:
            coord = tf.train.Coordinator()
            threads = self.load_model(sess, coord, model_dir)

            try:
                step = 0
                while step < num_iter and not coord.should_stop():
                    preds, targets = sess.run([evals_int, self.targets])
                    preds[preds == 0] = -1
                    seq_len_ind = np.sign(targets)
                    preds = preds * seq_len_ind
                    hits += (preds == 1).sum()
                    misses += (preds == -1).sum()
                    step += 1
            except Exception as e:
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads)

        print('Total Hits : {}\nTotal Misses : {}'.format(hits, misses))
        print('Accuracy : {:f}'.format(float(hits) / (hits + misses)))


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
