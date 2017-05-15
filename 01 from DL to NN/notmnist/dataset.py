from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

from notmnist.settings import data_root, image_size

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

#from notmnist.data_normalize_pickle import image_size

def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels

def dataset_pickle_file(fname):
    return os.path.join(data_root, fname)

def createDataset():

    train_size = 200000
    valid_size = 10000
    test_size = 10000

    from notmnist.data_normalize_pickle import train_datasets, test_datasets

    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
        train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    def randomize(dataset, labels):
      permutation = np.random.permutation(labels.shape[0])
      shuffled_dataset = dataset[permutation,:,:]
      shuffled_labels = labels[permutation]
      return shuffled_dataset, shuffled_labels

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

    ds = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    return ds

def saveDataset(ds, fname):
    fname = dataset_pickle_file(fname)
    try:
      f = open(fname, 'wb')
      pickle.dump(ds, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to', fname, ':', e)
      raise


train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
    None, None, None, None, None, None

def loadDataset(fname='notMNIST.pickle', load_globals=True):
    global train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
    ds = pickle.load(open(dataset_pickle_file(fname), 'rb'))
    if load_globals:
        train_dataset = ds['train_dataset']
        train_labels = ds['train_labels']
        valid_dataset = ds['valid_dataset']
        valid_labels = ds['valid_labels']
        test_dataset = ds['test_dataset']
        test_labels = ds['test_labels']
    return ds

def printDsetShapes(ds):
    train_dataset = ds['train_dataset']
    train_labels = ds['train_labels']
    valid_dataset = ds['valid_dataset']
    valid_labels = ds['valid_labels']
    test_dataset = ds['test_dataset']
    test_labels = ds['test_labels']
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

def printDsetStats(fname):
    statinfo = os.stat(dataset_pickle_file(fname))
    print('Compressed pickle size:', statinfo.st_size)

if __name__ == '__main__':
    saveDataset(createDataset(), 'notMNIST.pickle')