import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell

class InvertingLSTM():

    def __init__(self, seqSize, alphabetSize, batchSize, networkSize,
                        learningRate=0.01, depth=1, dropout=0.5):
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
        self.learningRate = learningRate
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
            self.rawBatchEOS = self.__addEOSchars(self.rawBatch)
            # conseq. of adding EOS char (to alphabet and sequences)
            self.alphabetSize += 1; self.seqSize += 1
            self.inputBatch = tf.one_hot(self.rawBatchEOS, self.alphabetSize, name='input_batch')
        with tf.name_scope('correct_output'):
            self.correctOutBatch = tf.reverse(self.inputBatch, [1], name='correct_output')

    def __addEOSchars(self, rawBatch):
        '''
        Add End-Of-Sequence chars (char indices) to the end of each input sequence.
        The EOS index will be self.alphabetSize (last letter + 1), and
            self.alphabetSize will be increased by 1
        :return:
        '''
        paddings = [[0, 0], [0, 1]]
        rawBatch = tf.pad(rawBatch, paddings, constant_values=self.alphabetSize)
        return rawBatch

    def __buildNetwork(self):
        def buildLSTMCell():
            def cell():
                c = BasicLSTMCell(num_units=self.networkSize)
                c = DropoutWrapper(c, output_keep_prob=self.dropout)
                return c

            if self.depth > 1:
                cell = MultiRNNCell([cell() for _ in range(self.depth)])
            else:
                cell = cell()
            return cell
        with tf.name_scope('coding_network'), tf.variable_scope('coding'):
            #with tf.variable_scope("model", reuse=True):
            ccell = buildLSTMCell()
            initState = ccell.zero_state(self.batchSize, tf.float32)
            self.cout, self.cstate = \
                tf.nn.dynamic_rnn(ccell, self.inputBatch, initial_state=initState)
        with tf.name_scope('decoding_network'), tf.variable_scope('decoding'):
            dcell = buildLSTMCell()
            # all zeros for now, but it can be passed self.cout as the first element
            self.decodeInput = tf.zeros((self.batchSize, self.seqSize, self.alphabetSize),
                                        dtype=tf.float32)
            # take output, ignore state
            self.dout, _ = tf.nn.dynamic_rnn(dcell, self.decodeInput, initial_state=self.cstate)
            self.doutFlat = tf.reshape(self.dout, [-1, self.networkSize], name='output_flat')

    def __buildLoss(self):
        '''
        Build loss function based on matching the output
        of the decoding network and the inverted input characters
        '''
        with tf.name_scope('predictions'):
            self.softmaxW = tf.Variable(tf.truncated_normal(
                                 (self.networkSize, self.alphabetSize), stddev=0.1),
                                 name='softmax_w')
            self.softmaxB = tf.Variable(tf.zeros(self.alphabetSize), name='softmax_b')
            self.doutLogits = tf.matmul(self.doutFlat, self.softmaxW)+self.softmaxB
            self.predictions = tf.nn.softmax(self.doutLogits, name='predictions')

        with tf.name_scope('loss'):
            self.outputReshaped = tf.reshape(self.correctOutBatch, self.doutLogits.get_shape(),
                                        name='output_reshaped')
            self.loss = tf.nn.softmax_cross_entropy_with_logits(
                logits= self.doutLogits, labels=self.outputReshaped, name='loss'
            )
            self.cost = tf.reduce_mean(self.loss, name='cost')

    def __buildOptimizer(self):
        with tf.name_scope('optimizer'):
            tvars = tf.trainable_variables()
            # todo see structure of gradients
            grad = tf.gradients(self.loss, tvars)
            grad, _ = tf.clip_by_global_norm(grad, 5)
            # todo use avg. clipping
            # grad, _ = tf.clip_by_average_norm(grad)
            self.optimizer = tf.train.AdamOptimizer(self.learningRate)
            self.optimizeOp = self.optimizer.apply_gradients(zip(grad, tvars))

    def evalInputs(self, inBatch):
        with tf.Session() as sess:
            inp, out = sess.run([self.inputBatch, self.correctOutBatch], feed_dict={self.rawBatch: inBatch})
            printt(inp); printt(out)

    def evalNetwork(self, inBatch):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            cout, cstate, dout = sess.run([self.cout, self.cstate, self.dout],
                                    feed_dict={self.rawBatch: inBatch})
            printt(cout); printt(cstate); printt(dout)
            print(tf.trainable_variables())

    def evalOutput(self, inBatch):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            dout, doutFlat = sess.run([self.dout, self.doutFlat],
                                    feed_dict={self.rawBatch: inBatch})
            printt(dout); printt(doutFlat)

    def evalLoss(self, inBatch):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            res = sess.run([self.predictions, self.outputReshaped, self.loss],
                                    feed_dict={self.rawBatch: inBatch})
            for v in res:
                print(v)

    def evalOptimizer(self, inBatch):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss, _ = sess.run([self.loss, self.optimizeOp],
                                   feed_dict={self.rawBatch: inBatch})
            printt(loss)
            loss, _ = sess.run([self.loss, self.optimizeOp],
                                   feed_dict={self.rawBatch: inBatch})
            printt(loss)

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
    # model.evalInputs(batch)
    # model.evalNetwork(batch)
    # model.evalOutput(batch)
    # model.evalLoss(batch)
    # model.evalOptimizer(batch)

def testTrain(seqSize=10, batchSize=128, networkSize=30):
    # init network
    # create train / test split
    # create batch generator on train
    # for each batch
    #   convert batch chars to char indices, add EOF symbol
    #   run optimizer op
    #   calculate loss on train batch
    #   calculate loss on test set
    # calculate loss on valid set
    from assignment6_lstm.batch_generator import BatchGenerator, batches2string
    from assignment6_lstm.dataset import validTrainSplit, vocabulary_size, char2id
    def indexString(s): return [ char2id(c) for c in s ]
    valid, train = validTrainSplit()
    batchgen = BatchGenerator(train, num_unrollings=seqSize, batch_size=batchSize)
    model = InvertingLSTM(seqSize, alphabetSize=vocabulary_size, batchSize=batchSize,
                          networkSize=networkSize, learningRate=0.001)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(200):
            inBatches = batches2string(batchgen.next())
            # print(type(inBatch), inBatch)
            # s = inBatch[0]
            # print(type(s), s)
            inBatch = [ indexString(s) for s in inBatches ]
            #print(inBatch)
            opt, cost, loss = sess.run([model.optimizeOp, model.cost, model.loss],
                                   feed_dict={model.rawBatch: inBatch})
            #print(i)
            print(i, cost)
            #print(loss)



if __name__ == '__main__':
    #test()
    testTrain()