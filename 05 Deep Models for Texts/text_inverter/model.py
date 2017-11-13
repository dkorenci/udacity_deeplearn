import tensorflow as tf

class InvertingLSTM():

    def __init__(self, seqSize, depth, alphabetSize, batchSize):
        self.seqSize, self.depth = seqSize, depth
        self.alphabetSize, self.batchSize = alphabetSize, batchSize
        self.__buildInputs()
        self.__buildNetwork()
        self.__buildLoss()
        self.__buildOptimizer()

    def __buildInputs(self):
        '''
        Define placeholder for input batch and derived input and expected output variables.
        '''
        with tf.name_scope('inputs'):
            self.rawBatch = tf.placeholder(tf.int32, (self.batchSize, self.seqSize), name='raw_batch')
            self.inputBatch = tf.one_hot(self.rawBatch, self.alphabetSize, name='input_batch')
        with tf.name_scope('correct_output'):
            self.correctOutBatch = tf.reverse(self.inputBatch, [1], name='correct_output')

    def evalInputs(self, inBatch):
        with tf.Session() as sess:
            inp, out = sess.run([self.inputBatch, self.correctOutBatch], feed_dict={self.rawBatch:inBatch})
            print(type(inp))
            print(inp)
            print(type(out))
            print(out)

    def __buildNetwork(self): pass

    def __buildLoss(self): pass

    def __buildOptimizer(self): pass

def test():
    model = InvertingLSTM(4, 2, 3, 3)
    batch = [
        [ 0, 2, 1, 1 ],
        [ 1, 2, 0, 2 ],
        [ 2, 0, 1, 0 ]
    ]
    model.evalInputs(batch)

if __name__ == '__main__':
    test()