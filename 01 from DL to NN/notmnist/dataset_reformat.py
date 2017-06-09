from notmnist.dataset import loadDataset, saveDataset, printDsetShapes

from notmnist.settings import image_size, num_labels

import numpy as np

def reformat1d(dataset, labels):
  '''
  Flatten last two dimensions (holding image pixels) from 2d array to 1d array
  Turn lables from integers to 1-hot encoding.
  :return:
  '''
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def reformatDataset(ds, reformatter):
    new = {}
    new['train_dataset'], new['train_labels'] = reformatter(ds['train_dataset'], ds['train_labels'])
    new['valid_dataset'], new['valid_labels'] = reformatter(ds['valid_dataset'], ds['valid_labels'])
    new['test_dataset'], new['test_labels'] = reformatter(ds['test_dataset'], ds['test_labels'])
    return new

def reformatConv(dataset, labels, image_size = 28, num_labels = 10, num_channels = 1):
    '''
    Reformat MNIST dataset for convolutional networks, turning images to 3d
        arrays by adding additional 'channel' dimension.
    '''
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

if __name__ == '__main__':
    ds = loadDataset()
    #ds = loadDataset('notMNIST_reformatted_1d_images.pickle')
    #printDsetShapes(ds)
    # nds = reformatDataset(ds, reformatConv)
    # printDsetShapes(nds)
    # saveDataset(nds, 'notMNIST_reformatted_conv.pickle')