from __future__ import division
from __future__ import print_function

import os
import json
import numpy

from tensorflow.python.platform import gfile
from utilities import deleteFiles, writeToFile

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_CPADS = '_CPADS'
_CPADE = '_CPADE'
_SEP = '_S_'
_START_VOCAB = [_PAD, _GO, _EOS]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def build_aspell():
    lexicon = []
    with open('./data/aspell.dict') as dfile:
        for line in dfile:
            lexicon.append(line.strip().lower().decode('utf-8'))
    return lexicon
aspell = build_aspell()


def get_training_data(data_dir):
    training_files = [
        'train_data.json',
        # 'clear_tweets.json'
    ]

    all_samples = []
    for jfile in training_files:
        with open(os.path.join(data_dir, jfile)) as ifi:
            samples = json.load(ifi)
        all_samples.extend(samples)

    return all_samples


def valid_token(token):
    try:
        token.encode('ascii')
    except UnicodeEncodeError:
        return False
    else:
        if token.startswith(('#', '@', 'http')):
            return False
        elif not token.isalnum():
            return False
        return True


def context_window(l, ngram):
    assert (ngram % 2) == 1
    assert ngram >= 1

    lpadded = ngram // 2 * [_CPADS] + l + ngram // 2 * [_CPADE]
    windows = [lpadded[i:(i + ngram)] for i in range(len(l))]

    assert len(windows) == len(l)
    return windows


def convert_format(window):
    spaced = []
    for token in window:
        if token == _CPADS:
            spaced.append(_CPADS)
        elif token == _CPADE:
            spaced.append(_CPADE)
        elif not valid_token(token.lower()):
            spaced.append(_UNK)
        else:
            spaced.append(' '.join(list(token.lower())))
    return spaced


def prepare_train_files(data_dir, train_portion=0.9, reuse=False):

    en_train = os.path.join(data_dir, 'train.en')
    fr_train = os.path.join(data_dir, 'train.fr')
    en_dev = os.path.join(data_dir, 'dev.en')
    fr_dev = os.path.join(data_dir, 'dev.fr')
    en_test = os.path.join(data_dir, 'test.en')
    fr_test = os.path.join(data_dir, 'test.fr')

    exist = [os.path.exists(f) for f in [
        en_train, fr_train, en_dev, fr_dev, en_test, fr_test
    ]]
    if reuse and all(exist):
        return (os.path.join(data_dir, 'train'), os.path.join(data_dir, 'dev'),
                os.path.join(data_dir, 'test'))

    samples = get_training_data(data_dir)

    deleteFiles([en_train, fr_train, en_dev, fr_dev, en_test, fr_test])

    ngram = 3
    num_samples = len(samples)
    num_train_samples = int(numpy.round(num_samples * train_portion))
    sep = ' {} '.format(_SEP)
    # Training Set
    for sample, n in zip(samples, range(num_train_samples)):
        for in_win, out in zip(context_window(sample['input'], ngram),
                               sample['output']):
            inp, out = in_win[ngram // 2].lower(), out.lower()
            if not valid_token(inp):
                continue
            # if inp in aspell:
            #     continue

            writeToFile(en_train, sep.join(convert_format(in_win)) + '\n')
            writeToFile(fr_train, ' '.join(list(out.replace(' ', '_')))
                        + '\n')

    # Dev Set
    for sample in samples[num_train_samples:]:
        for in_win, out in zip(context_window(sample['input'], ngram),
                               sample['output']):
            inp, out = in_win[ngram // 2].lower(), out.lower()
            if not valid_token(inp):
                continue
            # if inp in aspell:
            #     continue

            writeToFile(en_dev, sep.join(convert_format(in_win)) + '\n')
            writeToFile(fr_dev, ' '.join(list(out.replace(' ', '_')))
                        + '\n')

    # Test Set
    with open(os.path.join(data_dir, 'test_truth.json')) as ifi:
        samples = json.load(ifi)

    for sample in samples:
        for in_win, out in zip(context_window(sample['input'], ngram),
                               sample['output']):
            inp, out = in_win[ngram // 2].lower(), out.lower()
            if not valid_token(inp):
                continue
            if inp in aspell:
                continue

            writeToFile(en_test, sep.join(convert_format(in_win)) + '\n')
            writeToFile(fr_test, ' '.join(list(out.replace(' ', '_')))
                        + '\n')

    return (os.path.join(data_dir, 'train'), os.path.join(data_dir, 'dev'),
            os.path.join(data_dir, 'test'))


def create_vocabulary(vocab_path, data_path, reuse=False):
    if reuse and os.path.exists(vocab_path):
        return
    deleteFiles([vocab_path])
    vocab = {}
    with open(data_path) as f:
        for line in f:
            for token in line.split():
                if token in vocab:
                    vocab[token] += 1
                else:
                    vocab[token] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    writeToFile(vocab_path, '\n'.join(vocab_list))


def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary):
    tokens = sentence.split()
    return [vocabulary.get(w) for w in tokens]


def data_to_token_ids(data_path, target_path, vocab_path, reuse=False):
    if reuse and os.path.exists(target_path):
        return
    deleteFiles([target_path])
    vocab, _ = initialize_vocabulary(vocab_path)
    with gfile.GFile(data_path, mode="r") as data_file:
        with gfile.GFile(target_path, mode="w") as tokens_file:
            for line in data_file:
                token_ids = sentence_to_token_ids(line, vocab)
                tokens_file.write(" ".join([str(tok) for tok in token_ids])
                                  + "\n")


def prepare_data(data_dir, reuse=False):

    train_path, dev_path, test_path = prepare_train_files(data_dir,
                                                          reuse=reuse)

    fr_vocab_path = os.path.join(data_dir, "vocab.fr")
    en_vocab_path = os.path.join(data_dir, "vocab.en")
    create_vocabulary(fr_vocab_path, train_path + ".fr", reuse)
    create_vocabulary(en_vocab_path, train_path + ".en", reuse)

    fr_train_ids_path = train_path + ".ids.fr"
    en_train_ids_path = train_path + ".ids.en"
    data_to_token_ids(train_path + ".fr", fr_train_ids_path, fr_vocab_path, reuse)
    data_to_token_ids(train_path + ".en", en_train_ids_path, en_vocab_path, reuse)

    fr_dev_ids_path = dev_path + ".ids.fr"
    en_dev_ids_path = dev_path + ".ids.en"
    data_to_token_ids(dev_path + ".fr", fr_dev_ids_path, fr_vocab_path, reuse)
    data_to_token_ids(dev_path + ".en", en_dev_ids_path, en_vocab_path, reuse)

    fr_test_ids_path = test_path + ".ids.fr"
    en_test_ids_path = test_path + ".ids.en"
    data_to_token_ids(test_path + ".fr", fr_test_ids_path, fr_vocab_path, reuse)
    data_to_token_ids(test_path + ".en", en_test_ids_path, en_vocab_path, reuse)

    return (en_train_ids_path, fr_train_ids_path,
            en_dev_ids_path, fr_dev_ids_path,
            en_test_ids_path, fr_test_ids_path,
            en_vocab_path, fr_vocab_path)
