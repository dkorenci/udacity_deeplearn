from matplotlib import pylab
from sklearn.manifold import TSNE
from assignment5_word2vec.skipgram import train_skipgram, reverse_dictionary
from assignment5_word2vec.cbow import train_cbow

def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()

def reduce(embeddings):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    return tsne.fit_transform(embeddings)

def plotSkipgram(num_points = 400):
    emb = train_skipgram(train_steps=200001, window_size=2)
    emb2d = reduce(emb[1:num_points + 1, :])
    words = [reverse_dictionary[i] for i in range(1, num_points + 1)]
    plot(emb2d, words)

def plotCbow(num_points = 400):
    emb = train_cbow(train_steps=100001, window_size=2)
    emb2d = reduce(emb[1:num_points + 1, :])
    words = [reverse_dictionary[i] for i in range(1, num_points + 1)]
    plot(emb2d, words)

if __name__ == '__main__':
    plotSkipgram()
    #plotCbow()