import os
import json
import codecs
from contextlib import contextmanager
# import traceback as tb

NEW_LINE = '\n'
BUFFER_MAX = 5000

PATHS = {
    'root': '/home/ankur/devbench/tf-deep/seq_norm_main/',
}

VARS = {

    # This defines the extent of the context used for each input.
    # So if the input sequence is [x, y, z, w] and ngam=3, then the first input
    # to the encoder will be [_CPADS, x, y]; the second input to the encoder
    # will be [x, y, z]; the third [y, z, w] and so on. Please check
    # dataUtils.py for more info on 'padding start' (_CPADS) and 'padding end'
    # (_CPADE)
    'ngram': 3,

    # Size of the RNN layer
    'cell_size': 1024,

    # Minibatch Size
    'batch_size': 64,

    # number of RNN layers
    'num_layers': 3,

    # the global max of the norm of the gradient used for clipping during
    # backpropagation
    'max_gradient_norm': 5.0,

    'learning_rate': 0.0005,

    'learning_rate_decay_factor': 0.99,

    # no. of gpus that can be used separately for training different models
    # This should only be changed if GPUs are either added or removed from the
    # server.
    'num_gpus': 4,

    # number of shuffled files created for training.
    # This increases the randomness as the number goes higher.
    'num_shuffled_files': 3,
}


def to_unicode(text, encoding='utf-8'):
    # provided that the text is not already decoded.
    if not isinstance(text, unicode):
        text = unicode(text, encoding)
    return text


@contextmanager
def ignored(*exceptions):
    try:
        yield
    except exceptions:
        pass


# File I/O

def writeToFile(filename, line, mode='a'):
    """
    - mode can have either of 'w' or 'a' as its value
    - this function is not responsible for printing '\n' character
    """
    with codecs.open(filename, mode, encoding='utf-8') as ofile:
        ofile.write(line)


def deleteFiles(file_list, warn=True):
    if not isinstance(file_list, list):
        file_list = [file_list]
    for fname in file_list:
        with ignored(OSError):
            if warn:
                print "Deleting `{}` if it exists".format(fname)
            os.remove(fname)


# JSON I/O

def loadJSON(filename):
    with open(filename) as ifi:
        json_tree = json.load(ifi)
    return json_tree


def saveJSON(obj, filename):
    with open(filename, 'w') as ofi:
        json.dump(obj, ofi)
