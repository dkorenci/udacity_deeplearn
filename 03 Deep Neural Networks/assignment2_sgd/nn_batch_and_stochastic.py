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

from assignment2_sgd.logreg_batch_graddescent import accuracy

exampleWidth = image_size * image_size
def buildAndTrainModel(hiddenSize=100, learnRate=0.5, learnRateDecay=1.0, optimizeSteps = 1000, stochastic=False,
                       batchSize=128, trainSubsetSize=100000, reportEvery=100):
    graph = tf.Graph()
    global train_dataset, train_labels
    # truncate train dataset, only effective from non-stochastic training
    if trainSubsetSize:
        train_dataset = train_dataset[:trainSubsetSize]
        train_labels = train_labels[:trainSubsetSize]
    ## build graph
    with graph.as_default():
        # setup model
        if stochastic:
            train = tf.placeholder(np.float32, (batchSize, exampleWidth))
            trainLabels = tf.placeholder(np.float32, (batchSize, num_labels))
        else:
            train = tf.constant(train_dataset)
            trainLabels = tf.constant(train_labels)
        weights1 = tf.Variable(tf.truncated_normal((exampleWidth, hiddenSize)))
        bias1 = tf.Variable(tf.zeros(hiddenSize))
        weights2 = tf.Variable(tf.truncated_normal((hiddenSize, num_labels)))
        bias2 = tf.Variable(tf.zeros(num_labels))
        def nnOutput(input, raw = False):
            ''' create a variable representing network output for given input variable '''
            out = tf.nn.relu(tf.matmul(input, weights1)+bias1)
            out = tf.matmul(out, weights2)+bias2
            if not raw: out = tf.nn.softmax(out)
            return out
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=trainLabels, logits=nnOutput(train, raw=True)))
        optimizer = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)
        # setup classification performance indicators
        trainOutput = nnOutput(train)
        valid, test =  tf.constant(valid_dataset), tf.constant(test_dataset)
        validOutput, testOutput = nnOutput(valid), nnOutput(test)
    ## run model optimization
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
            _, train_res, valid_res, test_res = \
                session.run([optimizer, trainOutput, validOutput, testOutput],
                                                        feed_dict=feeds)
            if s % reportEvery == 0:
                print('step %d: train_acc %.4f, valid_acc %.4f' % \
                        (s, accuracy(train_res, trlabels), accuracy(valid_res, valid_labels)))
        print('TEST accuracy: %.4f' % accuracy(test_res, test_labels))

if __name__ == '__main__':
    #buildAndTrainModel(stochastic=True, hiddenSize=200, learnRate=0.5, optimizeSteps=1000)
    #buildAndTrainModel(stochastic=True, hiddenSize=200, learnRate=0.25, optimizeSteps=1000)
    buildAndTrainModel(stochastic=True, hiddenSize=1000, learnRate=0.01, optimizeSteps=1000, batchSize=1000)