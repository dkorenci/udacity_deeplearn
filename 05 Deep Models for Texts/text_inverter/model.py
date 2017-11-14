import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell

class InvertingLSTM():

    def __init__(self, seqSize, alphabetSize, batchSize, networkSize, depth=1, dropout=0.5):
        '''
        :param seqSize: lenght of sequences the net will be trained on
        :param depth: depth of LSTM, number of stacked cells
        :param alphabetSize: equiv. to the size of the char vector
        :param batchSize:
        :param networkSize: num. hidden units in the network
        :param dropout: rate of the dropout
        '''
        self.seqSize, self.depth = seqSize, depth
        self.alphabetSize, self.batchSize = alphabetSize, batchSize
        self.networkSize = networkSize; self.dropout = dropout
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

    def __buildNetwork(self):
        ccell = self.__buildCodingCell()
        initState = ccell.zero_state(self.batchSize, tf.float32)
        self.cout, self.cstate = \
            tf.nn.dynamic_rnn(ccell, self.inputBatch, initial_state=initState)
        dcell = self.__buildDecodingLSTM()

    def __buildCodingCell(self):
        def cell():
            c = BasicLSTMCell(num_units=self.networkSize)
            c = DropoutWrapper(c, output_keep_prob=self.dropout)
            return c
        if self.depth > 1: cell = MultiRNNCell([cell() for _ in range(self.depth)])
        else: cell = cell()
        return cell

    def __buildDecodingLSTM(self): pass


    def evalInputs(self, inBatch):
        with tf.Session() as sess:
            inp, out = sess.run([self.inputBatch, self.correctOutBatch], feed_dict={self.rawBatch: inBatch})
            printt(inp); printt(out)

    def evalCoding(self, inBatch):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            cout, cstate = sess.run([self.cout, self.cstate],
                                    feed_dict={self.rawBatch: inBatch})
            printt(cout); printt(cstate)

    def __buildLoss(self): pass

    def __buildOptimizer(self): pass

def printt(t):
    ''' Print tensor '''
    print(type(t))
    print(t)

def test():
    model = InvertingLSTM(4, 3, 3, 5)
    batch = [
        [ 0, 2, 1, 1 ],
        [ 1, 2, 0, 2 ],
        [ 2, 0, 1, 0 ]
    ]
    model.evalInputs(batch)
    model.evalCoding(batch)

if __name__ == '__main__':
    test()