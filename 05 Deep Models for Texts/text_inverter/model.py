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
        with tf.name_scope('coding_network'), tf.variable_scope('coding'):
            #with tf.variable_scope("model", reuse=True):
            ccell = self.__buildLSTMCell()
            initState = ccell.zero_state(self.batchSize, tf.float32)
            self.cout, self.cstate = \
                tf.nn.dynamic_rnn(ccell, self.inputBatch, initial_state=initState)
        with tf.name_scope('decoding_network'), tf.variable_scope('decoding'):
            dcell = self.__buildLSTMCell()
            # all zeros for now, but it can be passed self.cout as the first element
            self.decodeInput = tf.zeros((self.batchSize, self.seqSize, self.alphabetSize),
                                        dtype=tf.float32)
            self.dout, self.dstate = \
                tf.nn.dynamic_rnn(dcell, self.decodeInput, initial_state=self.cstate)

    def __buildLSTMCell(self):
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

    def evalNetwork(self, inBatch):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            cout, cstate, dout, dstate = sess.run([self.cout, self.cstate,
                                     self.dout, self.dstate],
                                    feed_dict={self.rawBatch: inBatch})
            printt(cout); printt(cstate)
            printt(dout); printt(dstate)
            print(tf.trainable_variables())

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
    model.evalNetwork(batch)

if __name__ == '__main__':
    test()