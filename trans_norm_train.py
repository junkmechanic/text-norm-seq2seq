from glob import glob
import tensorflow as tf
from utilities import loadJSON

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

learning_rate = 0.0005


def main(_):
    if FLAGS.test:
        batch_size = 1
        forward_only = True
        num_test_examples = sum([len(loadJSON(tfile)) for tfile in test_files])
    else:
        batch_size = 64
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
            cell_size=1024,
            num_layers=3,
            max_gradient_norm=5.0,
            learning_rate=learning_rate,
            learning_rate_decay_factor=0.99,
            forward_only=forward_only,
            predict=predict,
            model_dir='/home/ankur/devbench/tf-deep/seq_norm_main/train/',
            name='transNormModel_mixed_3L_1024U',
            eos_idx=eos_idx,
        )

    if not FLAGS.test:
        model.learn()
        # model.learn(new_learning_rate=learning_rate, reset_global_step=True)
    else:
        model.test(num_test_examples)

if __name__ == "__main__":
    tf.app.run()
