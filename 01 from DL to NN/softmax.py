"""Softmax."""

import numpy as np

scores1 = [3.0, 1.0, 0.2]
scores2 = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])
scores=scores2

def softmax_bruteForce(x):
    """Compute softmax values for each sets of scores in x."""
    if not isinstance(x, np.ndarray): x = np.array(x)
    smax = lambda x: np.exp(x) / np.exp(x).sum()
    if len(x.shape) == 1: return smax(x)
    else:
        r = np.empty(x.shape)
        for i in range(x.shape[1]):
            r[:, i] = smax(x[:, i])
        return r

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print(softmax(scores))

#Plot softmax curves
if False:
    import matplotlib.pyplot as plt
    x = np.arange(-2.0, 6.0, 0.1)
    scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
    print(scores.shape)
    plt.plot(x, softmax(scores).T, linewidth=2)
    plt.show()
