from __future__ import print_function
import collections
import math
import numpy as np
import random
import tensorflow as tf
from six.moves import range

from assignment5_word2vec.dataset import vocabulary_size, loadDataset

data, count, dictionary, reverse_dictionary = loadDataset()

data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

def test_batch():
    print('data:', [reverse_dictionary[di] for di in data[:20]])
    for num_skips, skip_window in [(2, 1), (4, 2)]:
        batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
        print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
        print('    batch:', [reverse_dictionary[bi] for bi in batch])
        print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

def train_skipgram(learnRate=1.0, num_sampled = 64, train_steps = 100001, window_size = 1):
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    num_skips = 2  # How many times to reuse an input to generate a label.
    # We pick a random validation set to sample nearest neighbors. here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.array(random.sample(range(valid_window), valid_size))
     # Number of negative examples to sample.

    graph = tf.Graph()

    with graph.as_default(): #, tf.device('/cpu:0'):
        # Input data.
        train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Variables.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        softmax_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Model.
        # Look up embeddings for inputs.
        embed = tf.nn.embedding_lookup(embeddings, train_dataset)
        # Compute the softmax loss, using a sample of the negative labels each time.
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                       labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

        # Optimizer.
        # Note: The optimizer will optimize the softmax_weights AND the embeddings.
        # This is because the embeddings are defined as a variable quantity and the
        # optimizer's `minimize` method will by default modify all variable quantities
        # that contribute to the tensor it is passed.
        # See docs on `tf.train.Optimizer.minimize()` for more details.
        optimizer = tf.train.AdagradOptimizer(learning_rate=learnRate).minimize(loss)

        # Compute the similarity between minibatch examples and all embeddings.
        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print('Initialized')
      average_loss = 0
      for step in range(train_steps):
        batch_data, batch_labels = generate_batch(
          batch_size, num_skips, window_size)
        feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
          if step > 0:
            average_loss = average_loss / 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step %d: %f' % (step, average_loss))
          average_loss = 0
        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
          sim = similarity.eval()
          for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8 # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k+1]
            log = 'Nearest to %s:' % valid_word
            for k in range(top_k):
              close_word = reverse_dictionary[nearest[k]]
              log = '%s %s,' % (log, close_word)
            print(log)
      final_embeddings = normalized_embeddings.eval()
    return final_embeddings

if __name__ == '__main__':
    #test_batch()
    #train_skipgram(learnRate=0.1)
    train_skipgram(num_sampled=128)