from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle

import numpy as np
from time import time

from notmnist.settings import pickle_file, image_size

def loadDataset():
    return pickle.load(open(pickle_file, 'rb'))


def tf(ds):
    '''
    Transform 3d dataset with 2d image matrices in rows,
    into 2d dataset with flattened images in rows.
    '''
    return np.reshape(ds, (ds.shape[0], image_size*image_size))


def trainTest():
    ds = loadDataset()
    logReg = LogisticRegression(multi_class='multinomial', solver='sag', n_jobs=3)
    t0 = time()
    #logReg.fit(tf(ds['train_dataset']), ds['train_labels'])
    logReg.fit(tf(ds['test_dataset']), ds['test_labels'])
    te = time() - t0
    print('%.4f seconds' % te)
    print('%.4f minutes' % (te/60))
    print(logReg.score(tf(ds['valid_dataset']), ds['valid_labels']))

if __name__ == '__main__':
    trainTest()
