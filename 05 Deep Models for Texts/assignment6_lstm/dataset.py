import zipfile
import tensorflow as tf
import string

from assignment5_word2vec.dataset import maybe_download

def read_data(filename):
  '''
  Read data from a zipfile, return as single string.
  '''
  with zipfile.ZipFile(filename) as f:
    name = f.namelist()[0]
    data = tf.compat.as_str(f.read(name))
  return data

def getData():
    return read_data(maybe_download('text8.zip', 31344016))

def validTrainSplit(valid_size = 1000, verbose=False):
    '''
    Split dataset into valid and train.
    '''
    text = getData()
    valid_text = text[:valid_size]
    train_text = text[valid_size:]
    train_size = len(train_text)
    if verbose:
        print(train_size, train_text[:64])
        print(valid_size, valid_text[:64])
    return valid_text, train_text

vocabulary_size = len(string.ascii_lowercase) + 1  # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ': return 0
    else:
        print('Unexpected character: %s' % char)
        return 0

def id2char(dictid):
    if dictid > 0: return chr(dictid + first_letter - 1)
    else: return ' '

if __name__ == '__main__':
    validTrainSplit(verbose=True)
    print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
    print(id2char(1), id2char(26), id2char(0))