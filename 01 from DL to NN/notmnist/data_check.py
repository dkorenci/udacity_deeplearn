from IPython.display import display, Image

import glob, random
files = [fname for fname in
         glob.iglob('/data/code/udacity_deeplearn/01 from DL to NN/notmnist/notMNIST_small/**/*.png',
                    recursive=True)]
random.shuffle(files)

for i in range(20):
    display(Image(filename=files[i]))