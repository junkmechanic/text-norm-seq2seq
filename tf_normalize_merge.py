from collections import namedtuple
from eval import evaluate
from data_utils import context_window, valid_token, build_aspell
from utilities import loadJSON, saveJSON
from diff_errors import perform_diff
from multiprocessing import Pool


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


def get_prediction(pred_cache, index, position):
    return pred_cache[index]['prediction'][position]


def postPred(word, aspell):
    word = cleanupToken(word)

    if word.endswith('in'):
        possible = '%sg' % word
        if possible in aspell:
            word = possible

    return word


sources = build_sources()
aspell = build_aspell()
pred_cache = loadJSON('./data/test_out_only_oov.json')
pred_cache = {d['index']: {k: d[k] for k in d if k != 'index'}
              for d in pred_cache}


def concurrent_normalize(sample):
    ngram = 3
    sample['prediction'] = []
    sample['flags'] = []
    for i, win in enumerate(context_window(sample['input'], ngram)):
        token = win[ngram // 2].lower()
        if not valid_token(token):
            sample['prediction'].append(token)
            sample['flags'].append('ignored')
            continue
        overridden = ruleBasedPrediction(token, sources.vocabplus)
        if overridden:
            sample['prediction'].append(overridden)
            sample['flags'].append('overridden')
            continue
        ruled_token = ruleBasedPrediction(token, sources.subdict)
        if ruled_token:
            sample['prediction'].append(ruled_token)
            sample['flags'].append('ruled')
            continue
        if token in aspell:
            sample['prediction'].append(token)
            sample['flags'].append('vocab')
            continue
        baselined = ruleBasedPrediction(token, sources.baseline)
        if baselined:
            sample['prediction'].append(baselined)
            sample['flags'].append('baselined')
        else:
            # Use the seq2seq model to predict the out token
            nn_pred = get_prediction(pred_cache, sample['index'], i)
            better_pred = postPred(nn_pred, aspell)
            sample['prediction'].append(better_pred)
            sample['flags'].append('predicted')
    return sample


if __name__ == '__main__':
    samples = loadJSON('./data/test_truth.json')
    pool = Pool(62)
    samples = pool.map(concurrent_normalize, samples)
    saveJSON(samples, './data/test_out.json')
    evaluate(samples)
    perform_diff()
