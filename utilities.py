import os
import json
import codecs
from contextlib import contextmanager
# import traceback as tb

NEW_LINE = '\n'
BUFFER_MAX = 5000

PATHS = {
    'root': '/home/ankur/devbench/tf-deep/seq_norm_main/'
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
