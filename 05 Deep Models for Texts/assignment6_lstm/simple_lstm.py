import numpy as np
import tensorflow as tf

from assignment6_lstm.dataset import vocabulary_size
from assignment6_lstm.probability_utils import logprob, sample, random_distribution


def simpleLstm(num_nodes=64, num_unrollings=10, batch_size=64, num_steps = 7001):
    '''
    :param num_nodes: number of hidden units in (various components of) the model
    :param num_unrollings: max. length to which the cell is replicated
    :param batch_size: size of a stochastic gradient descent batch
    :return:
    '''
    graph = tf.Graph()
    with graph.as_default():
        # Parameters:
        # Input gate: input, previous output, and bias.
        ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        ib = tf.Variable(tf.zeros([1, num_nodes]))
        # Forget gate: input, previous output, and bias.
        fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        fb = tf.Variable(tf.zeros([1, num_nodes]))
        # Memory cell: input, state and bias.
        cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        cb = tf.Variable(tf.zeros([1, num_nodes]))
        # Output gate: input, previous output, and bias.
        ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        ob = tf.Variable(tf.zeros([1, num_nodes]))
        # Variables saving state across unrollings.
        saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        # Classifier weights and biases.
        w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
        b = tf.Variable(tf.zeros([vocabulary_size]))


        # Definition of the cell computation.
        def lstm_cell(i, o, state):
            """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
            Note that in this formulation, we omit the various connections between the
            previous state and the gates."""
            input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
            forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
            update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
            state = forget_gate * state + input_gate * tf.tanh(update)
            output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
            return output_gate * tf.tanh(state), state


        # Input data.
        train_data = list()
        for _ in range(num_unrollings + 1):
            train_data.append(
                tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))
        train_inputs = train_data[:num_unrollings]
        train_labels = train_data[1:]  # labels are inputs shifted by one time step.

        # Unrolled LSTM loop.
        outputs = list()
        output = saved_output
        state = saved_state
        for i in train_inputs:
            output, state = lstm_cell(i, output, state)
            outputs.append(output)

        # State saving across unrollings.
        with tf.control_dependencies([saved_output.assign(output),
                                      saved_state.assign(state)]):
            # Classifier.
            logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.concat(train_labels, 0), logits=logits))

        # Optimizer.
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            10.0, global_step, 5000, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step)

        # Predictions.
        train_prediction = tf.nn.softmax(logits)

        # Sampling and validation eval: batch 1, no unrolling.
        sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
        saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
        saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
        reset_sample_state = tf.group(
            saved_sample_output.assign(tf.zeros([1, num_nodes])),
            saved_sample_state.assign(tf.zeros([1, num_nodes])))
        sample_output, sample_state = lstm_cell(
            sample_input, saved_sample_output, saved_sample_state)
        with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                      saved_sample_state.assign(sample_state)]):
            sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

    # learning params
    summary_frequency = 100
    # prepare batch generator
    from assignment6_lstm.dataset import validTrainSplit
    from assignment6_lstm.batch_generator import BatchGenerator, characters
    valid_text, train_text = validTrainSplit()
    valid_size = len(valid_text)
    train_batches = BatchGenerator(train_text, batch_size=batch_size, num_unrollings=num_unrollings)
    valid_batches = BatchGenerator(valid_text, 1, 1)

    # run learning
    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print('Initialized')
      mean_loss = 0
      for step in range(num_steps):
        batches = train_batches.next()
        feed_dict = dict()
        for i in range(num_unrollings + 1):
          feed_dict[train_data[i]] = batches[i]
        _, l, predictions, lr = session.run(
          [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        if step % summary_frequency == 0:
          if step > 0:
            mean_loss = mean_loss / summary_frequency
          # The mean loss is an estimate of the loss over the last few batches.
          print(
            'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
          mean_loss = 0
          labels = np.concatenate(list(batches)[1:])
          print('Minibatch perplexity: %.2f' % float(
            np.exp(logprob(predictions, labels))))
          if step % (summary_frequency * 10) == 0:
            # Generate some samples.
            print('=' * 80)
            for _ in range(5):
              feed = sample(random_distribution())
              sentence = characters(feed)[0]
              reset_sample_state.run()
              for _ in range(79):
                prediction = sample_prediction.eval({sample_input: feed})
                feed = sample(prediction)
                sentence += characters(feed)[0]
              print(sentence)
            print('=' * 80)
          # Measure validation set perplexity.
          reset_sample_state.run()
          valid_logprob = 0
          for _ in range(valid_size):
            b = valid_batches.next()
            predictions = sample_prediction.eval({sample_input: b[0]})
            valid_logprob = valid_logprob + logprob(predictions, b[1])
          print('Validation set perplexity: %.2f' % float(np.exp(
            valid_logprob / valid_size)))

if __name__ == '__main__':
    simpleLstm(num_nodes=128, batch_size=128, num_unrollings=50, num_steps=20001)