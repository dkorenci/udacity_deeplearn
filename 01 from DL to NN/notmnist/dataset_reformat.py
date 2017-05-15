from notmnist.dataset import loadDataset, saveDataset, printDsetShapes

from notmnist.settings import image_size, num_labels

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def reformattedDataset(ds):
    new = {}
    new['train_dataset'], new['train_labels'] = reformat(ds['train_dataset'], ds['train_labels'])
    new['valid_dataset'], new['valid_labels'] = reformat(ds['valid_dataset'], ds['valid_labels'])
    new['test_dataset'], new['test_labels'] = reformat(ds['test_dataset'], ds['test_labels'])
    return new

if __name__ == '__main__':
    #ds = loadDataset()
    ds = loadDataset('notMNIST_reformatted.pickle')
    printDsetShapes(ds)
    #nds = reformattedDataset(ds)
    #printDsetShapes(nds)
    #saveDataset(nds, 'notMNIST_reformatted.pickle')