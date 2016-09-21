import numpy as np
import tensorflow as tf
from eval import evaluate
from trans_norm_model import TransNormModel
from dataUtils import DataSource
from utilities import loadJSON, saveJSON


def normalize(samples):
    ngram = 3
    sep = ' _S_ '
    batch_size = 1
    forward_only = predict = True

    dsource = DataSource(reuse=True)
    aspell = dsource.aspell
    vocab, rev_vocab = DataSource.initialize_vocabulary(dsource.vocab_path)
    eos_idx = vocab['_EOS']

    inp_seq_batch = tf.placeholder(tf.int64, shape=[batch_size, None])
    out_seq_batch = tf.placeholder(tf.int64, shape=[batch_size, None])
    targets_batch = tf.placeholder(tf.int64, shape=[batch_size, None])
    model_dir = '/home/ankur/devbench/tf-deep/seq_norm_main/train/'
    model = TransNormModel(
        inp_seq_batch,
        out_seq_batch,
        targets_batch,
        dsource.vocab_size,
        batch_size,
        cell_size=1024,
        num_layers=3,
        max_gradient_norm=5.0,
        learning_rate=0.0005,
        learning_rate_decay_factor=0.99,
        forward_only=forward_only,
        predict=predict,
        model_dir=model_dir,
        eos_idx=eos_idx,
    )
    sess = tf.Session(config=model.sess_conf)
    model.load_model(sess, None, model.model_dir)

    def predict_word(in_win):
        token_ids = DataSource.sentence_to_token_ids(
            sep.join(DataSource.convert_format(win)),
            vocab
        )
        trigger = [vocab['_GO']]
        dummy_target = [vocab['_GO']]
        outputs = sess.run(model.predicted_word,
                           feed_dict={
                               inp_seq_batch.name: [token_ids],
                               out_seq_batch.name: [trigger],
                               targets_batch.name: [dummy_target]
                           })[1].reshape((-1))
        if eos_idx in outputs:
            outputs = outputs[:np.where(outputs == eos_idx)[0][0]]
        out_token = ''.join([rev_vocab[out] for out in outputs])
        return out_token.replace('_', ' ')

    count = 0
    total, ignored, ruled, predicted = (0, 0, 0, 0)
    print 'total no. of samples : ', len(samples)
    for sample in samples:
        sample['prediction'] = []
        sample['flags'] = []
        for win in DataSource.context_window(sample['input'], ngram):
            total += 1
            token = win[ngram // 2].lower()
            if not DataSource.valid_token(token):
                sample['prediction'].append(token)
                sample['flags'].append('ignored')
                ignored += 1
                continue
            if token in aspell:
                sample['prediction'].append(token)
                sample['flags'].append('vocab')
                ignored += 1
            else:
                # Use the seq2seq model to predict the out token
                sample['prediction'].append(predict_word(win))
                sample['flags'].append('predicted')
                predicted += 1
        count += 1
        if count % 100 == 0:
            print 'completed {} samples'.format(count)
    sess.close()
    print 'Total: {}\nIgnored: {}\nRuled: {}\nPredicted: {}'.format(
        total, ignored, ruled, predicted
    )


if __name__ == '__main__':
    samples = loadJSON('./data/test_truth.json')
    normalize(samples)
    saveJSON(samples, './data/test_out_only_oov.json')
    evaluate(samples, './data/norm_errors_only_oov.json')
