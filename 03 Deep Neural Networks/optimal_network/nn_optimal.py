'''
Further performance optimizations of previous networks by adding
more layers and learning rate decay.
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes

from notmnist.dataset import loadDataset
from notmnist.settings import *

ds = loadDataset('notMNIST_reformatted.pickle')
train_dataset = ds['train_dataset']
train_labels = ds['train_labels']
valid_dataset = ds['valid_dataset']
valid_labels = ds['valid_labels']
test_dataset = ds['test_dataset']
test_labels = ds['test_labels']
exampleWidth = image_size * image_size

import attr

@attr.s
class MultilayerNN():
    ''' Fully connected multilayer NN. '''
    input = attr.ib()
    inputSize = attr.ib()
    outputSize = attr.ib()
    hiddenLayers = attr.ib([10]) # sizes of hidden layers
    dropout = attr.ib(0) # dropout rate, or 0/None for no dropout

    def __attrs_post_init__(self):
        self.__initGraph()

    def __initGraph(self):
        '''Create weights and bias tensors '''
        self.numHidden = len(self.hiddenLayers)
        self.numLayers = self.numHidden+2
        self.weights, self.bias = [None] * self.numLayers, [None] * self.numLayers
        inSize = self.inputSize
        for i in range(1, self.numLayers):
            output = (i == self.numLayers-1) # weather we're at output layer
            outSize = self.hiddenLayers[i-1] if not output else self.outputSize
            self.weights[i] = tf.Variable(tf.truncated_normal((inSize, outSize)))
            self.bias[i] = tf.Variable(tf.zeros(outSize))
            inSize = outSize

    def __drop(self, f):
        ''' apply dropout to a function or pass through '''
        return tf.nn.dropout(f, self.dropout) if self.dropout else f

    def out(self, input, train=False, activation=tf.nn.softmax):
        '''
        Build function that represents the network output with given input
        :param train: if True, dropout will be applied
        :return:
        '''
        layers = [None] * self.numLayers
        layers[0] = input
        for i in range(1, self.numLayers):
            output = (i == self.numLayers-1) # weather we're at output layer
            if train: layers[i-1] = self.__drop(layers[i-1])
            layers[i] = tf.matmul(layers[i - 1], self.weights[i]) + self.bias[i]
            if not output:
                layers[i] = tf.nn.relu(layers[i])
        outLayer = layers[self.numLayers-1]
        if activation is None: return outLayer
        else: return activation(outLayer)

def buildAndTrainModel(layers=[500], learnRate=0.01, momentum=0.95, dropout=0.5, decay=0.99, decayStart=100,
                       optimizeSteps = 1000, stochastic=True, batchSize=128, trainSubsetSize=100000, reportEvery=100):
    graph = tf.Graph()
    global train_dataset, train_labels
    # truncate train dataset, only effective for non-stochastic training
    if trainSubsetSize:
        train_dataset = train_dataset[:trainSubsetSize]
        train_labels = train_labels[:trainSubsetSize]
    ## build graph
    with graph.as_default():
        # tf.add_check_numerics_ops()
        # setup model
        trainFull = tf.constant(train_dataset)
        if stochastic:
            train = tf.placeholder(np.float32, (batchSize, exampleWidth))
            trainLabels = tf.placeholder(np.float32, (batchSize, num_labels))
        else:
            train = trainFull
            trainLabels = tf.constant(train_labels)
        network = MultilayerNN(input=train, inputSize=exampleWidth, outputSize=num_labels,
                               hiddenLayers = layers, dropout=dropout)
        trainOutput = network.out(train, True, activation=None)
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=trainLabels,
                                                        logits=trainOutput))
        # setup learning rate decay and optmization
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learnRate, global_step, decay_steps=decayStart,
                                                   decay_rate=decay, staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step)
        # setup classification performance indicators
        trainOutput = network.out(train)
        fullTrainOutput = network.out(trainFull)
        valid, test = tf.constant(valid_dataset), tf.constant(test_dataset)
        validOutput, testOutput = network.out(valid), network.out(test)
    ## run model optimization
    from assignment2_sgd.logreg_batch_graddescent import accuracy
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for s in range(optimizeSteps):
            if stochastic:
                start = (s*batchSize) % (train_dataset.shape[0] - batchSize)
                feeds = {train: train_dataset[start:start+batchSize],
                         trainLabels: train_labels[start:start+batchSize]}
                trlabels = train_labels[start:start+batchSize]
            else:
                feeds=None
                trlabels = train_labels
            # execute GDesc update
            session.run([optimizer], feed_dict=feeds)
            if s % reportEvery == 0:
                lrate, train_res, full_train_res, valid_res, test_res = \
                    session.run([learning_rate, trainOutput, fullTrainOutput, validOutput, testOutput],
                                feed_dict=feeds)
                print('learning rate %.4f' % lrate)
                print('step %d: batch_train_acc %.4f, full_train_acc %.4f, valid_acc %.4f, test_acc %.4f' % \
                        (s, accuracy(train_res, trlabels), accuracy(full_train_res, train_labels),
                            accuracy(valid_res, valid_labels), accuracy(test_res, test_labels)))
        print('TEST accuracy: %.4f' % accuracy(test_res, test_labels))

if __name__ == '__main__':
    buildAndTrainModel(layers=[200,200], batchSize=500, learnRate=0.00001, dropout=0.5,
                       momentum=0.8, decay=0.999, decayStart=10000, optimizeSteps=20000)