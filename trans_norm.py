# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time
import itertools
from datetime import datetime

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model

from utilities import deleteFiles, writeToFile


tf.app.flags.DEFINE_float("learning_rate", 0.00005, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 43, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 44, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_integer("patience", 20, "Patience")
tf.app.flags.DEFINE_boolean("reuse", True, "Reuse prepared data")

FLAGS = tf.app.flags.FLAGS

# Limit (hard) the amount of GPU memory to be used by a process during a
# session
config_all = tf.ConfigProto()
# config_all.gpu_options.per_process_gpu_memory_fraction=0.5

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# _buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
_buckets = [(10, 15), (20, 25), (40, 50)]


def set_vocab_size(vocab_path, lang='en'):
  with open(vocab_path) as ifi:
    vocab_size = len(ifi.readlines())
  FLAGS.__setattr__(lang + '_vocab_size', vocab_size)


def read_data(filename_queue):
  """
  This function reads the TFRecords file and returns a single sample tensor.
  The returned tensor will generally be added to a queue.
  """

  reader = tf.RecordReader()
  _, serialized_sample = reader.read(filename_queue)
  context_features, sequences = tf.parse_single_sequence_example(
      serialized_sample,
      sequence_features={
          'inp_seq': tf.FixedLenSequenceFeature((1,), tf.int64,
                                                allow_missing=False),
          'out_seq': tf.FixedLenSequenceFeature((1,), tf.int64,
                                                allow_missing=False),
      }
  )

  # since the SequenceExample is stored in at least a 2D array, the first
  # dimension being the length of the sequence and the second being the feature
  # size for each item in the sequence, we need to flatten it. This is because
  # in our case, the items are merely token ids.
  inp_seq = tf.reshape(sequences['inp_seq'], [-1])
  out_seq = tf.reshape(sequences['out_seq'], [-1])

  return inp_seq, out_seq


def prepare_data_queues(datasource, set_type):
  filenames = [getattr(datasource, set_type + '_path')]
  filename_queue = tf.train.string_input_producer(filenames)
  inp_seq, out_seq = read_data(filename_queue)

  # next is to generate the batch using tf.train.shuffle_batch
  inp_seq_batch, out_seq_batch = tf.train.shuffle_batch([inp_seq, out_seq],
                                                        FLAGS.batch_size,
                                                        dynamic_pad=True,
                                                        name="sequence_batch")


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  print(FLAGS.en_vocab_size, FLAGS.fr_vocab_size)
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.en_vocab_size, FLAGS.fr_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, use_lstm=True,
      forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("{} : Reading model parameters from {}".format(
        datetime.now().ctime(), ckpt.model_checkpoint_path))
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("{} : Created model with fresh parameters.".format(
        datetime.now().ctime()))
    session.run(tf.initialize_all_variables())
  return model


def train():
  print("Preparing data in %s" % FLAGS.data_dir)
  # change the reuse parameter if you want to build the data again
  en_train, fr_train, en_dev, fr_dev, en_test, fr_test, \
      en_vocab_path, fr_vocab_path = \
      data_utils.prepare_data(FLAGS.data_dir, reuse=FLAGS.reuse)

  set_vocab_size(en_vocab_path, 'en')
  set_vocab_size(fr_vocab_path, 'fr')

  with tf.Session(config=config_all) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(en_dev, fr_dev)
    train_set = read_data(en_train, fr_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    print('Bucket Sizes : {}'.format(train_bucket_sizes))
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    if not os.path.exists(FLAGS.train_dir):
      os.makedirs(FLAGS.train_dir)

    # Creating summaries for the parameters
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    for var in tf.trainable_variables():
      summaries.append(tf.histogram_summary(var.op.name, var))

    # Creating a summary writer
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
    summary_op = tf.merge_all_summaries()

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    history_ppxs = []
    bad_counter = 0
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.expm1(loss) if loss < 300 else float('inf')
        print ("%s : global step %d learning rate %.7f step-time %.2f "
               "perplexity %.9f" % (datetime.now().ctime(),
                                    model.global_step.eval(),
                                    model.learning_rate.eval(),
                                    step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        bucket_ppx = []
        for bucket_id in xrange(len(_buckets)):
          dev_batches = [[u for u in k if u is not None] for k in
                         itertools.izip_longest(
                             *[dev_set[bucket_id][i::FLAGS.batch_size]
                               for i in range(FLAGS.batch_size)])]
          for batch in dev_batches[:-1]:
            encoder_inputs, decoder_inputs, target_weights = model.prepare_batch(
                batch, bucket_id)
            _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True)
            eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            bucket_ppx.append(eval_ppx)
        dev_ppx = np.mean(bucket_ppx)
        print("  dev eval: perplexity %.5f" % (dev_ppx))
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, model.global_step.eval())
        history_ppxs.append(dev_ppx)
        if (len(history_ppxs) > FLAGS.patience and
           dev_ppx >= np.array(history_ppxs)[:-FLAGS.patience].min()):
          bad_counter += 1
          # if bad_counter > FLAGS.patience:
          #   print("Patience reached")
          #   break
        sys.stdout.flush()


def update_error_counts(in_seqs, out_seqs):
  # During the test_eval, send in buckets
  # During decoding, dissolve the buckets and send the entire input
  test_out = os.path.join(FLAGS.data_dir, 'test_errors.out')
  error_str = '{}\nIn: {}\nNorm: {}\nOut: {}\n\n'

  _, rev_en_vocab = data_utils.initialize_vocabulary(os.path.join(
      FLAGS.data_dir, 'vocab.en'))
  _, rev_fr_vocab = data_utils.initialize_vocabulary(os.path.join(
      FLAGS.data_dir, 'vocab.fr'))

  stats = {'R2W': 0, 'W2R': 0, 'W2W_C': 0, 'W2W_NC': 0}

  for in_seq, out_seq in zip(in_seqs, out_seqs):
    # inp = ''.join([rev_en_vocab[i] for i in in_seq[0]]).replace('_', ' ')
    inp = ''.join([rev_en_vocab[i] for i in in_seq[0]])
    inp = inp.split('_S_')[1]
    norm = ''.join([rev_fr_vocab[i] for i in in_seq[1][:-1]]).replace('_', ' ')
    out = ''.join([rev_fr_vocab[i] for i in out_seq]).replace('_', ' ')

    if inp == norm:
      if out != norm:
        stats['R2W'] += 1
        writeToFile(test_out, error_str.format('R2W', inp, norm, out))
    else:
      if out == norm:
        stats['W2R'] += 1
        # writeToFile(test_out, error_str.format('W2R', inp, norm, out))
      elif out == inp:
        stats['W2W_NC'] += 1
        writeToFile(test_out, error_str.format('W2W_NC', inp, norm, out))
      else:
        stats['W2W_C'] += 1
        writeToFile(test_out, error_str.format('W2W_C', inp, norm, out))
  return stats


def eval_test():
  tf.reset_default_graph()
  test_out = os.path.join(FLAGS.data_dir, 'test_errors.out')
  deleteFiles([test_out])
  stats = {'R2W': 0, 'W2R': 0, 'W2W_C': 0, 'W2W_NC': 0}
  # change the reuse parameter if you want to build the data again
  _, _, _, _, en_test, fr_test, _, _ = data_utils.prepare_data(FLAGS.data_dir,
                                                               reuse=FLAGS.reuse)
  with tf.Session(config=config_all) as sess:
    model = create_model(sess, True)
    test_set = read_data(en_test, fr_test)
    test_bucket_sizes = [len(test_set[b]) for b in range(len(_buckets))]
    print('Bucket Sizes : {}'.format(test_bucket_sizes))
    total_loss, num_batches = 0, 0

    for bucket_id in range(len(_buckets)):
      all_batches = ([u for u in k if u is not None] for k in
                     itertools.izip_longest(
                         *[test_set[bucket_id][i::FLAGS.batch_size]
                           for i in range(FLAGS.batch_size)]))
      for batch in all_batches:
        encoder_inputs, decoder_inputs, target_weights = model.prepare_batch(
            batch, bucket_id)
        # setting the model batch size in case it is smaller (would be for the
        # last batch in the bucket)
        model.batch_size = len(batch)
        _, eval_loss, logits = model.step(sess, encoder_inputs, decoder_inputs,
                                          target_weights, bucket_id, True)
        outputs = np.argmax(logits, axis=2).transpose()
        outseq = [out[:list(out).index(data_utils.EOS_ID)] for out in outputs
                  if data_utils.EOS_ID in out]
        stat_updates = update_error_counts(batch, outseq)
        stats = {k: stats[k] + v for k, v in stat_updates.items()}
        total_loss += math.exp(eval_loss)
        num_batches += 1
        # resetting the madel batch size
        model.batch_size = FLAGS.batch_size
    print("Loss over the test set : {}".format(total_loss / num_batches))
    print(stats)
    precision = stats['W2R'] / sum([stats['W2R'], stats['R2W'],
                                    stats['W2W_C']])
    recall = stats['W2R'] / sum([stats['W2R'], stats['W2W_NC'],
                                 stats['W2W_C']])
    f_m = (2 * precision * recall) / (precision + recall)
    print('P: {}\nR: {}\nF: {}'.format(precision, recall, f_m))


def decode(in_file, with_labels=True):
  with tf.Session(config=config_all) as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab.en")
    fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab.fr")
    en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def main(_):
  if FLAGS.decode:
    # decode()
    eval_test()
  else:
    train()
    # eval_test()

if __name__ == "__main__":
  tf.app.run()
