import os
import numpy as np
import tensorflow as tf
from collections import namedtuple
from eval import evaluate
from trans_norm_model import TransNormModel
from dataUtils import DataSource
from utilities import loadJSON, saveJSON, PATHS, VARS

tf.app.flags.DEFINE_integer("gpu", 0,
                            "Index of the GPU to be used for creating the "
                            "graph of the model")

FLAGS = tf.app.flags.FLAGS


def cleanupToken(token, rep_allowed=1):
    """
    rep_allowed : no. of repretitions allowed. For example:
        'sloooow' -> 'slow' for rep_allowed = 0
        'sloooow' -> 'sloow' for rep_allowed = 1
    """
    Extra = namedtuple('Extra', ['index', 'times'])
    extra_occurences = []
    prev = ''
    counter = 0
    for i in range(len(token)):
        if token[i] == prev:
            counter += 1
        else:
            if counter > rep_allowed:
                pos = i - (counter - rep_allowed)
                extra = counter - rep_allowed
                extra_occurences.append(Extra(index=pos, times=extra))
            counter = 0
            prev = token[i]
    if counter > rep_allowed:
        pos = i - (counter - rep_allowed)
        extra = counter - rep_allowed
        extra_occurences.append(Extra(index=pos, times=extra))
    new_token = list(token)
    deleted = 0
    for extra in extra_occurences:
        for _ in range(extra.times):
            del new_token[extra.index - deleted]
        deleted += extra.times
    return ''.join(new_token)


def build_sources():
    Sources = namedtuple('Sources', ['subdict', 'hbdict', 'utddict',
                                     'baseline', 'vocabplus'])
    hbdict = {}
    with open('./dict/hb.dict') as ifi:
        for line in ifi:
            noise, norm = map(str.strip, line.split('\t'))
            if noise in hbdict:
                hbdict[noise].append(norm)
            else:
                hbdict[noise] = [norm]
    utddict = {}
    with open('./dict/utd.dict') as ifi:
        for line in ifi:
            freq, token_str = line.split('\t')
            tokens = map(str.strip, token_str.split('|'))
            if tokens[0] in utddict:
                utddict[tokens[0]].extend(tokens[1:])
            else:
                utddict[tokens[0]] = tokens[1:]
    subdict = loadJSON('./dict/sub.dict')
    baseline = loadJSON('./dict/baseline_rules.json')
    vocabplus = loadJSON('./dict/override.dict')
    return Sources(subdict=subdict, hbdict=hbdict, utddict=utddict,
                   baseline=baseline, vocabplus=vocabplus)


def ruleBasedPrediction(token, source):
    if token in source:
        return source[token][0]


def postPred(word, aspell):
    word = cleanupToken(word)

    if word.endswith('in'):
        possible = '%sg' % word
        if possible in aspell:
            word = possible
    # this is too aggressive
    # elif re.match(r'.*(\w)(\1)$', word) and \
    #         len(re.match(r'.*(\w)(\1)$', word).groups()) == 2 and \
    #         word[:-1] in aspell:
    #     word = word[:-1]

    return word


def normalize(samples, gpu_device):
    ngram = VARS['ngram']
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
    model_dir = os.path.join(PATHS['root'] + 'train/')
    with tf.device(gpu_device):
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

    sources = build_sources()
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
            overridden = ruleBasedPrediction(token, sources.vocabplus)
            if overridden:
                sample['prediction'].append(overridden)
                sample['flags'].append('overridden')
                ruled += 1
                continue
            ruled_token = ruleBasedPrediction(token, sources.subdict)
            if ruled_token:
                sample['prediction'].append(ruled_token)
                sample['flags'].append('ruled')
                ruled += 1
                continue
            if token in aspell:
                sample['prediction'].append(token)
                sample['flags'].append('vocab')
                ignored += 1
                continue
            baselined = ruleBasedPrediction(token, sources.baseline)
            if baselined:
                sample['prediction'].append(baselined)
                sample['flags'].append('baselined')
                ruled += 1
            else:
                # Use the seq2seq model to predict the out token
                sample['prediction'].append(predict_word(win))
                sample['flags'].append('predicted')
                predicted += 1
        count += 1
        if count % 1000 == 0:
            print 'completed {} samples'.format(count)
    sess.close()
    print 'Total: {}\nIgnored: {}\nRuled: {}\nPredicted: {}'.format(
        total, ignored, ruled, predicted
    )


def main(_):
    if FLAGS.gpu < 0 or FLAGS.gpu > VARS['num_gpus'] - 1:
        raise ValueError("The index of the GPU should be between 0 and "
                         "{}".format(VARS['num_gpus'] - 1))
    else:
        gpu_device = '/gpu:{}'.format(FLAGS.gpu)
    samples = loadJSON('./data/test_truth.json')
    normalize(samples, gpu_device)
    saveJSON(samples, './data/test_out.json')
    evaluate(samples)


if __name__ == '__main__':
    tf.app.run()
