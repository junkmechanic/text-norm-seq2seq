# Copyright 2015 Google Inc. All Rights Reserved.
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
import json

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model

from utilities import deleteFiles, writeToFile


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 42, "French vocabulary size.")
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

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.en_vocab_size, FLAGS.fr_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      num_samples=0, forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train():
  print("Preparing data in %s" % FLAGS.data_dir)
  en_train, fr_train, en_dev, fr_dev, en_test, fr_test, _, _ = \
      data_utils.prepare_data(FLAGS.data_dir)

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(en_dev, fr_dev)
    test_set = read_data(en_test, fr_test)
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
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.5f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        print('training loss : {}'.format(loss))
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
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              test_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  test eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        for bucket_id in xrange(len(_buckets) - 1):
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          bucket_ppx.append(eval_ppx)
          print("  dev eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        dev_ppx = np.mean(bucket_ppx)
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
    inp = ''.join([rev_en_vocab[i] for i in in_seq[0]]).replace('_', ' ')
    norm = ''.join([rev_fr_vocab[i] for i in in_seq[1][:-1]]).replace('_', ' ')
    out = ''.join([rev_fr_vocab[i] for i in out_seq]).replace('_', ' ')

    if inp == norm:
      if out != norm:
        stats['R2W'] += 1
        writeToFile(test_out, error_str.format('R2W', inp, norm, out))
    else:
      if out == norm:
        stats['W2R'] += 1
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
  _, _, _, _, en_test, fr_test, _, _ = data_utils.prepare_data(FLAGS.data_dir)
  with tf.Session() as sess:
    model = create_model(sess, True)
    test_set = read_data(en_test, fr_test)
    total_loss, num_batches = 0, 0
    for bucket_id in range(len(_buckets)):
      for encoder_inputs, decoder_inputs, target_weights, batch_size in \
          model.get_all_batches(test_set, bucket_id):
        # setting the model batch size in case it is smaller (would be for the
        # last batch in the bucket)
        model.batch_size = batch_size
        _, eval_loss, logits = model.step(sess, encoder_inputs, decoder_inputs,
                                          target_weights, bucket_id, True)
        outputs = np.argmax(logits, axis=2).transpose()
        outseq = [out[:list(out).index(data_utils.EOS_ID)] for out in outputs
                  if data_utils.EOS_ID in out]
        stat_updates = update_error_counts(test_set[bucket_id], outseq)
        stats = {k: stats[k] + v for k, v in stat_updates.items()}
        total_loss += math.exp(eval_loss)
        num_batches += 1
        # resetting the madel batch size
        model.batch_size = FLAGS.batch_size
    print("Loss over the test set : {}".format(total_loss / num_batches))
    print(stats)


def decode(in_file, with_labels=True):
  with tf.Session() as sess:
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
      token_ids = data_utils.sentence_to_token_ids(sentence, en_vocab)
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
      print(" ".join([rev_fr_vocab[output] for output in outputs]))
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
