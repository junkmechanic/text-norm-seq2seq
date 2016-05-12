# import numpy as np
# import tensorflow as tf
from collections import namedtuple
from eval import evaluate
# from trans_norm import create_model, _buckets
from data_utils import context_window, convert_format, valid_token, \
    build_aspell, sentence_to_token_ids, initialize_vocabulary, EOS_ID
from utilities import loadJSON, saveJSON
from diff_errors import perform_diff


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
    # if token in sources.subdict:
    #     return sources.subdict[token][0]
    # elif token in sources.baseline:
    #     if len(sources.baseline[token]) < 2:
    #         return sources.baseline[token][0]


def get_prediction(pred_cache, index, position):
    return pred_cache[index]['prediction'][position]


def postPred(word, aspell):
    word = cleanupToken(word)

    if word.endswith('in'):
        possible = '%sg' % word
        if possible in aspell:
            word = possible

    return word


def normalize(samples):
    ngram = 3
    # sep = ' _S_ '
    sources = build_sources()
    aspell = build_aspell()

    # Load seq2seq model
    # sess = tf.Session()
    # model = create_model(sess, True)
    # model.batch_size = 1
    # en_vocab, _ = initialize_vocabulary('./data/vocab.en')
    # _, rev_fr_vocab = initialize_vocabulary('./data/vocab.fr')

    # all the predictions
    pred_cache = loadJSON('./data/test_out_only_oov.json')
    pred_cache = {d['index']: {k: d[k] for k in d if k != 'index'}
                  for d in pred_cache}

    # def predict_word(in_win):
    #     token_ids = sentence_to_token_ids(sep.join(convert_format(win)),
    #                                       en_vocab)
    #     bucket_id = min([b for b in xrange(len(_buckets))
    #                      if _buckets[b][0] > len(token_ids)])
    #     encoder_inputs, decoder_inputs, target_weights = model.get_batch(
    #         {bucket_id: [(token_ids, [])]}, bucket_id)
    #     _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
    #                                      target_weights, bucket_id, True)
    #     outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    #     if EOS_ID in outputs:
    #         outputs = outputs[:outputs.index(EOS_ID)]
    #     out_token = ''.join([rev_fr_vocab[out] for out in outputs])
    #     return out_token.replace('_', ' ')

    count = 0
    total, ignored, ruled, predicted = (0, 0, 0, 0)
    print 'total no. of samples : ', len(samples)
    for sample in samples:
        sample['prediction'] = []
        sample['flags'] = []
        for i, win in enumerate(context_window(sample['input'], ngram)):
            total += 1
            token = win[ngram // 2].lower()
            if not valid_token(token):
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
                nn_pred = get_prediction(pred_cache, sample['index'], i)
                better_pred = postPred(nn_pred, aspell)
                sample['prediction'].append(better_pred)
                sample['flags'].append('predicted')
                predicted += 1
        count += 1
        if count % 1000 == 0:
            print 'completed {} samples'.format(count)
    # sess.close()
    print 'Total: {}\nIgnored: {}\nRuled: {}\nPredicted: {}'.format(
        total, ignored, ruled, predicted
    )


if __name__ == '__main__':
    samples = loadJSON('./data/test_truth.json')
    normalize(samples)
    saveJSON(samples, './data/test_out.json')
    evaluate(samples)
    perform_diff()
