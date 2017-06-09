url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '/data/code/udacity_deeplearn/01 from DL to NN/notmnist'

import os
pickle_file = os.path.join(data_root, 'notMNIST.pickle')
image_size = 28  # Pixel width and height.
num_labels = 10
pixel_depth = 255.0  # Number of levels per pixel.
num_channels = 1