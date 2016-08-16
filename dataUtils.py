from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random
from itertools import izip

import tensorflow as tf
from tensorflow.python.platform import gfile
from utilities import deleteFiles, writeToFile, saveJSON, loadJSON

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_CPADS = '_CPADS'
_CPADE = '_CPADE'
_SEP = '_S_'

# Including padding in the beginning will ensure that index '0' in the
# vocabulary is not occupied by and other character. This is in line with the
# dynamic padding done during creation of batches later.
_START_VOCAB = [_PAD]


class DataSource:
    """
    This is the base class for all data sources. It accepts the dir in which
    the processed data will be stored.

    Each entry in the file_list should be the complete path. No path inference
    is done on the file_list.

    `train_ratio` defines how much of the data in the training files will be
    used for training. The rest will be included in the test data.

    `dev_ratio` defines how much of the train data will be separated for
    development set.

    `use_vocab` lets you provide a path to a vocabulary. If not specified, a
    vocabulary will be created from the data provided in its own 'data_dir'.
    The value of `use_vocab` should be the dir in which the vocab files exist.
    For example `project_root/data/default`

    `filter_vocab_test/train` are flags to indicate whether clean words would
    be filtered out (not considered) while building the dataset.

    The members that are defined later are:
        aspell

    """
    def __init__(self, name="default", data_dir='./data', train_files=None,
                 test_files=None, train_ratio=1.0, dev_ratio=0.1, ngram=3,
                 use_vocab=None, seed=2016, filter_vocab_train=False,
                 filter_vocab_test=True, reuse=False):

        self.name = name
        self.data_dir = os.path.join(data_dir, name)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.train_files = train_files if isinstance(train_files, list) \
            else [train_files] if train_files else []
        self.test_files = test_files if isinstance(test_files, list) \
            else [test_files] if test_files else []

        self.seed = seed
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.ngram = ngram
        self.vocab_path = os.path.join(use_vocab, 'vocab') if use_vocab \
            else os.path.join(self.data_dir, 'vocab')
        self.filter_vocab_train = filter_vocab_train
        self.filter_vocab_test = filter_vocab_test
        self.reuse = reuse

        self.build_aspell()
        self.train_path = os.path.join(self.data_dir, 'train')
        self.dev_path = os.path.join(self.data_dir, 'dev')
        self.test_path = os.path.join(self.data_dir, 'test')
        self.prepare_samples()
        if not use_vocab:
            self.vocab_size = self.build_vocabulary()
        else:
            with open(self.vocab_path) as ifi:
                self.vocab_size = len(ifi.readlines())
        self.build_ids()
        self.save_state()

    def build_aspell(self):
        self.aspell = []
        with open('./dict/aspell.dict') as dfile:
            for line in dfile:
                self.aspell.append(line.strip().lower().decode('utf-8'))

    def save_state(self):
        attributes = {k: v for k, v in vars(self).items() if k != 'aspell'}
        saveJSON(attributes, os.path.join(self.data_dir, 'state.json'))

    def build_vocabulary(self):
        if self.reuse and os.path.exists(self.vocab_path):
            with open(self.vocab_path) as ifi:
                return len(ifi.readlines())
        deleteFiles([self.vocab_path])
        vocab = {}
        for data_path in [self.train_path + '.inp', self.train_path + '.out']:
            with open(data_path) as f:
                for line in f:
                    for token in line.split():
                        vocab[token] = vocab.get(token, 0) + 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        writeToFile(self.vocab_path, '\n'.join(vocab_list))
        return len(vocab_list)

    def build_ids(self):

        def iter_over(inp_filename, out_filename):
            with open(inp_filename) as inp, open(out_filename) as out:
                for inp_seq, out_seq in izip(inp, out):
                    yield inp_seq.strip(), out_seq.strip()

        def format_sequence(sequence):
            return tf.train.FeatureList(
                feature=[
                    tf.train.Feature(int64_list=tf.train.Int64List(value=[s]))
                    for s in sequence
                ]
            )

        vocab, _ = self.initialize_vocabulary(self.vocab_path)
        for set_type in [self.train_path, self.dev_path, self.test_path]:
            target_path = set_type + '.ids.bin'
            if self.reuse and os.path.exists(target_path):
                return
            deleteFiles([target_path])
            writer = tf.python_io.TFRecordWriter(target_path)
            for in_seq, out_seq in iter_over(set_type + '.inp',
                                             set_type + '.out'):
                in_tokens = self.sentence_to_token_ids(in_seq, vocab)
                out_tokens = self.sentence_to_token_ids(out_seq, vocab)
                example = tf.train.SequenceExample(
                    feature_lists=tf.train.FeatureLists(
                        feature_list={
                            'inp_seq': format_sequence(in_tokens),
                            'out_seq': format_sequence(out_tokens),
                        }))
                writer.write(example.SerializeToString())
            writer.close()

    def prepare_samples(self):
        sep = ' {} '.format(_SEP)
        random.seed(self.seed)
        train_samples = self.load_samples(self.train_files) \
            if self.train_files else []
        test_samples = self.load_samples(self.test_files) \
            if self.test_files else []
        num_samples = len(train_samples)
        shuffled_indices = range(num_samples)
        random.shuffle(shuffled_indices)

        num_trn_dev_samples = int(np.round(num_samples * self.train_ratio))
        num_dev_samples = int(np.round(num_trn_dev_samples * self.dev_ratio))
        num_train_samples = num_trn_dev_samples - num_dev_samples

        # Filenames
        inp_train = self.train_path + '.inp'
        out_train = self.train_path + '.out'
        inp_dev = self.dev_path + '.inp'
        out_dev = self.dev_path + '.out'
        inp_test = self.test_path + '.inp'
        out_test = self.test_path + '.out'

        exist = [os.path.exists(f) for f in [
            inp_train, out_train, inp_dev, out_dev, inp_test, out_test
        ]]
        if self.reuse and all(exist):
            return
        deleteFiles([inp_train, out_train, inp_dev, out_dev, inp_test,
                     out_test])

        # Training Set
        for idx in shuffled_indices[:num_train_samples]:
            for in_win, out in zip(
                self.context_window(train_samples[idx]['input'], self.ngram),
                train_samples[idx]['output']
            ):
                inp = in_win[self.ngram // 2].lower()
                out = [_GO] + list(out.lower().replace(' ', '_')) + [_EOS]
                # If the output is blank, then we ignore the sample since we do
                # not want to consider such cases for training. Example if the
                # input was ['b', 'cuz'] the output will be ['because', ''].
                # For now, the model is not being trained to recognize that.
                if not self.valid_token(inp) or len(out) == 0:
                    continue
                if self.filter_vocab_train and inp in self.aspell:
                    continue
                writeToFile(inp_train,
                            sep.join(self.convert_format(in_win)) + '\n')
                # The output sequence might contain space implying that the
                # normalized output was a set of two words. This space is
                # represented as an underscore (`_`) becasue space is used as a
                # delimited in the input format.
                # Later while decoding, the output of the model should be
                # processed to replace underscores with spaces.
                writeToFile(out_train, ' '.join(out) + '\n')

        # Dev Set
        for idx in shuffled_indices[num_train_samples:num_trn_dev_samples]:
            for in_win, out in zip(
                self.context_window(train_samples[idx]['input'], self.ngram),
                train_samples[idx]['output']
            ):
                inp = in_win[self.ngram // 2].lower()
                out = [_GO] + list(out.lower().replace(' ', '_')) + [_EOS]
                if not self.valid_token(inp) or len(out) == 0:
                    continue
                if self.filter_vocab_train and inp in self.aspell:
                    continue

                writeToFile(inp_dev, sep.join(self.convert_format(in_win))
                            + '\n')
                writeToFile(out_dev, ' '.join(out) + '\n')

        # Test Set
        test_samples.extend(train_samples[num_trn_dev_samples:])
        for sample in test_samples:
            for in_win, out in zip(
                self.context_window(sample['input'], self.ngram),
                sample['output']
            ):
                inp = in_win[self.ngram // 2].lower()
                out = [_GO] + list(out.lower().replace(' ', '_')) + [_EOS]
                if not self.valid_token(inp) or len(out) == 0:
                    continue
                if self.filter_vocab_test and inp in self.aspell:
                    continue

                writeToFile(inp_test, sep.join(self.convert_format(in_win))
                            + '\n')
                writeToFile(out_test, ' '.join(out) + '\n')

    @staticmethod
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

    @staticmethod
    def load_samples(file_list):
        # The file_list should have complete path of the files
        all_samples = []
        for filename in file_list:
            samples = loadJSON(filename)
            all_samples.extend(samples)
        return all_samples

    @staticmethod
    def context_window(l, ngram):
        assert (ngram % 2) == 1
        assert ngram >= 1

        lpadded = ngram // 2 * [_CPADS] + l + ngram // 2 * [_CPADE]
        windows = [lpadded[i:(i + ngram)] for i in range(len(l))]

        assert len(windows) == len(l)
        return windows

    @staticmethod
    def convert_format(window):
        spaced = []
        for token in window:
            if token == _CPADS:
                spaced.append(_CPADS)
            elif token == _CPADE:
                spaced.append(_CPADE)
            elif not DataSource.valid_token(token.lower()):
                spaced.append(_UNK)
            else:
                spaced.append(' '.join(list(token.lower())))
        return spaced

    @staticmethod
    def initialize_vocabulary(vocab_path):
        if gfile.Exists(vocab_path):
            rev_vocab = []
            with gfile.GFile(vocab_path, mode="r") as f:
                rev_vocab.extend(f.readlines())
            rev_vocab = [line.strip() for line in rev_vocab]
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
            return vocab, rev_vocab
        else:
            raise ValueError("Vocabulary file %s not found.", vocab_path)

    @staticmethod
    def sentence_to_token_ids(sentence, vocabulary):
        tokens = sentence.split()
        return [vocabulary.get(w) for w in tokens]
