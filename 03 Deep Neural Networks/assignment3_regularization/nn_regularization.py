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
def buildAndTrainModel(hiddenSize=100, learnRate=0.5, optimizeSteps = 1000, stochastic=False,
                       regWeight = None, dropout=False,
                       batchSize=128, trainSubsetSize=100000, reportEvery=100):
    graph = tf.Graph()
    global train_dataset, train_labels
    # truncate train dataset, only effective for non-stochastic training
    if trainSubsetSize:
        train_dataset = train_dataset[:trainSubsetSize]
        train_labels = train_labels[:trainSubsetSize]
    ## build graph
    with graph.as_default():
        # setup model
        trainFull = tf.constant(train_dataset)
        if stochastic:
            train = tf.placeholder(np.float32, (batchSize, exampleWidth))
            trainLabels = tf.placeholder(np.float32, (batchSize, num_labels))
        else:
            train = trainFull
            trainLabels = tf.constant(train_labels)
        weights1 = tf.Variable(tf.truncated_normal((exampleWidth, hiddenSize)))
        bias1 = tf.Variable(tf.zeros(hiddenSize))
        weights2 = tf.Variable(tf.truncated_normal((hiddenSize, num_labels)))
        bias2 = tf.Variable(tf.zeros(num_labels))
        def nnOutput(input, raw = False, drop=False):
            ''' create a variable representing network output for given input variable '''
            out = input
            out = tf.nn.dropout(out, 0.5) if drop else out
            out = tf.nn.relu(tf.matmul(out, weights1)+bias1)
            out = tf.nn.dropout(out, 0.5) if drop else out
            out = tf.matmul(out, weights2)+bias2
            if not raw: out = tf.nn.softmax(out)
            return out
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=trainLabels,
                                                        logits=nnOutput(train, raw=True, drop=dropout)))
        if regWeight: # create and add l2 regularization of weights to l2
            regW = tf.constant(regWeight)
            regTerm = regW * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2))
            loss = loss + regTerm
        optimizer = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)
        # setup classification performance indicators
        trainOutput = nnOutput(train)
        fullTrainOutput = nnOutput(trainFull)
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
            session.run([optimizer], feed_dict=feeds)
            if s % reportEvery == 0:
                train_res, full_train_res, valid_res, test_res = \
                    session.run([trainOutput, fullTrainOutput, validOutput, testOutput], feed_dict=feeds)
                print('step %d: batch_train_acc %.4f, full_train_acc %.4f, valid_acc %.4f, test_acc %.4f' % \
                        (s, accuracy(train_res, trlabels), accuracy(full_train_res, train_labels),
                            accuracy(valid_res, valid_labels), accuracy(test_res, test_labels)))
        print('TEST accuracy: %.4f' % accuracy(test_res, test_labels))

if __name__ == '__main__':
    buildAndTrainModel(stochastic=True, hiddenSize=2000, learnRate=0.5, optimizeSteps=301, dropout=True)
    # buildAndTrainModel(stochastic=False, hiddenSize=1500, learnRate=0.1,
    #                    optimizeSteps=100, reportEvery=5, trainSubsetSize=20000)
    #buildAndTrainModel(stochastic=True, hiddenSize=500, learnRate=0.25, optimizeSteps=400, dropout=True)
    #buildAndTrainModel(stochastic=True, hiddenSize=1000, learnRate=0.25, optimizeSteps=1000, batchSize=300)
