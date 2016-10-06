import os
from glob import glob
import tensorflow as tf
from utilities import loadJSON, PATHS, VARS

from dataUtils import DataSource
from trans_norm_model import TransNormModel

tf.app.flags.DEFINE_boolean("test", False,
                            "Set to True for testing the model.")

FLAGS = tf.app.flags.FLAGS

train_files = [
    './data/clear_tweets.json',
    './data/train_data.json',
]

test_files = [
    './data/test_truth.json',
]


def main(_):
    if FLAGS.test:
        batch_size = 1
        forward_only = True
        num_test_examples = sum([len(loadJSON(tfile)) for tfile in test_files])
    else:
        batch_size = VARS['batch_size']
        forward_only = False
    predict = False

    dsource = DataSource(
        train_files=train_files,
        name='mixed',
        dev_ratio=0.0,
        test_files=test_files,
        use_vocab='./data/default/',
        reuse=True,
        num_shuffled_files=3
    )
    vocab, _ = DataSource.initialize_vocabulary(dsource.vocab_path)
    eos_idx = vocab['_EOS']

    if not FLAGS.test:
        f_queue = tf.train.string_input_producer(glob(dsource.train_path +
                                                      '.ids.bin.*'))
    else:
        f_queue = tf.train.string_input_producer([dsource.test_path +
                                                  '.ids.bin'])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(f_queue)
    context_features, sequences = tf.parse_single_sequence_example(
        serialized_example,
        sequence_features={
            'inp_seq': tf.FixedLenSequenceFeature((1,), tf.int64,
                                                  allow_missing=False),
            'out_seq': tf.FixedLenSequenceFeature((1,), tf.int64,
                                                  allow_missing=False),
            'targets': tf.FixedLenSequenceFeature((1,), tf.int64,
                                                  allow_missing=False),
        }
    )
    inp_seq, out_seq, target_seq = (sequences['inp_seq'], sequences['out_seq'],
                                    sequences['targets'])
    inp_seq = tf.reshape(inp_seq, [-1])
    out_seq = tf.reshape(out_seq, [-1])
    target_seq = tf.reshape(target_seq, [-1])
    inp_seq_batch, out_seq_batch, targets_batch = tf.train.batch(
        [inp_seq, out_seq, target_seq],
        batch_size,
        dynamic_pad=True,
        name="sequence_batch"
    )

    with tf.device('/gpu:1'):
        model = TransNormModel(
            inp_seq_batch,
            out_seq_batch,
            targets_batch,
            dsource.vocab_size,
            batch_size,
            cell_size=VARS['cell_size'],
            num_layers=VARS['num_layers'],
            max_gradient_norm=VARS['max_gradient_norm'],
            learning_rate=VARS['learning_rate'],
            learning_rate_decay_factor=VARS['learning_rate_decay_factor'],
            forward_only=forward_only,
            predict=predict,
            model_dir=os.path.join(PATHS['root'] + 'train/'),
            name='transNormModel_mixed_3L_1024U',
            eos_idx=eos_idx,
        )

    if not FLAGS.test:
        model.learn()
        # the function learn() can also be used in the way called below with the
        # extra parameters. This is used when you want to coninue training a
        # model that has already been (pre-)trained in a different dataset.
        # model.learn(new_learning_rate=learning_rate, reset_global_step=True)
    else:
        model.test(num_test_examples)

if __name__ == "__main__":
    tf.app.run()
