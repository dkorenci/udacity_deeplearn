from assignment6_lstm.dataset import char2id, id2char, vocabulary_size, \
        validTrainSplit

import numpy as np

class BatchGenerator(object):
    def __init__(self, text, batch_size = 64, num_unrollings = 10):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]


def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s

def testGenerator():
    valid_text, train_text = validTrainSplit()
    train_batches = BatchGenerator(train_text)
    valid_batches = BatchGenerator(valid_text, 1, 1)
    print(batches2string(train_batches.next()))
    print(batches2string(train_batches.next()))
    print(batches2string(valid_batches.next()))
    print(batches2string(valid_batches.next()))

if __name__ == '__main__':
    testGenerator()