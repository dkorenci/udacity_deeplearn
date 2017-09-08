from __future__ import print_function
import collections
import os
import tensorflow as tf
import zipfile
from six.moves.urllib.request import urlretrieve

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  from os import path
  modulefolder = path.dirname(__file__)
  return path.join(modulefolder, filename)

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def get_data():
    '''
     Return dataset text as list of word strings.
    '''
    filename = 'text8.zip'
    maybe_download(filename, 31344016)
    return read_data(filename)

vocabulary_size=50000

def build_dataset(words, vocabulary_size):
    '''
    Keep most frequent vocabulary_size words, transform other words to 'UNK' (unknown) token.
    Build dictionary.
    :param words: list of words
    :return: list of dict indexes of words, word counts,
            dict (word 2 index), reverse dict
    '''
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    print(dictionary['UNK'])
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

def test_build():
    words = get_data()
    data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    del words

import pickle

def createSaveDataset():
    dset = build_dataset(get_data(), vocabulary_size)
    f = open('text8_processed.pickle', 'wb')
    pickle.dump(dset, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def loadDataset():
    dset = pickle.load(open('text8_processed.pickle', 'rb'))
    data, count, dictionary, reverse_dictionary = dset
    return data, count, dictionary, reverse_dictionary

if __name__ == '__main__':
    #print(len(get_data()))
    #test_build()
    createSaveDataset()