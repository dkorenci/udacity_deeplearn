from notmnist.dataset import loadDataset, saveDataset, printDsetShapes

from notmnist.settings import image_size, num_labels

import numpy as np

def reformat1d(dataset, labels):
  '''
  Flatten last two dimensions (holding image pixels) from 2d array to 1d array
  :return:
  '''
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def reformatted1Ddataset(ds):
    new = {}
    new['train_dataset'], new['train_labels'] = reformat1d(ds['train_dataset'], ds['train_labels'])
    new['valid_dataset'], new['valid_labels'] = reformat1d(ds['valid_dataset'], ds['valid_labels'])
    new['test_dataset'], new['test_labels'] = reformat1d(ds['test_dataset'], ds['test_labels'])
    return new

if __name__ == '__main__':
    #ds = loadDataset()
    ds = loadDataset('notMNIST_reformatted_1d_images.pickle')
    printDsetShapes(ds)
    #nds = reformattedDataset(ds)
    #printDsetShapes(nds)
    #saveDataset(nds, 'notMNIST_reformatted.pickle')