'''
Toy Bars data, with K=6 topics and vocabulary size 144.
3 horizontal bars, and 3 vertical bars.

Generated via the standard mixture model
see BagOfWordsData.CreateToyDataFromMixtureModel for details.
'''
from builtins import *
import numpy as np
import bnpy

def make_bars_topics(V, K, fracMassOnTopic=0.95, PRNG=np.random):
    ''' Create parameters of each topics distribution over words

    Args
    ---------
    V : int vocab size
    K : int number of topics
    fracMassOnTopic : fraction of probability mass for "on-topic" words
    PRNG : random number generator (for reproducibility)

    Returns
    ---------
    topics : 2D array, K x V
        positive reals, each row sums to one
    '''
    sqrtV = int(np.sqrt(V))
    BarWidth = sqrtV / (K / 2)  # number of consecutive words in each bar
    B = V / (K / 2)  # total number of "on topic" words in each bar

    topics = np.zeros((K, V))
    # Make horizontal bars
    for k in range(K / 2):
        wordIDs = list(range(B * k, B * (k + 1)))
        topics[k, wordIDs] = 1.0

    # Make vertical bars
    for k in range(K / 2):
        wordIDs = list()
        for b in range(sqrtV):
            start = b * sqrtV + k * BarWidth
            wordIDs.extend(list(range(start, start + BarWidth)))
        topics[K / 2 + k, wordIDs] = 1.0

    # Add smoothing mass to all entries in "topics"
    #  instead of picking this value out of thin air, instead,
    #  set so 95% of the mass of each topic is on the "on-topic" bar words
    #  if s is the smoothing mass added, and B is num "on topic" words, then
    # fracMassOnTopic = (1 + s) * B / ( (1+s)*B + s*(V-B) ), and we solve for
    # s
    smoothMass = (1 - fracMassOnTopic) / (fracMassOnTopic * V - B) * B
    topics += (2 * smoothMass) * PRNG.rand(K, V)

    # Ensure each row of topics is a probability vector
    for k in range(K):
        topics[k, :] /= np.sum(topics[k, :])

    assert np.sum(topics[0, :B]) > fracMassOnTopic - 0.05
    assert np.sum(topics[1, B:2 * B]) > fracMassOnTopic - 0.05
    assert np.sum(topics[-1, wordIDs]) > fracMassOnTopic - 0.05
    return topics

if __name__ == '__main__':
    K = 6  # Number of topics
    V = 144  # Vocabulary Size
    SEED = 42
    PRNG = np.random.RandomState(SEED)
    Defaults = dict()
    Defaults['seed'] = SEED
    Defaults['nDocTotal'] = 2000
    Defaults['nWordsPerDoc'] = 100
    proba_K = np.asarray([2.0, 1.8, 1.6, 1.4, 1.2, 1.0])
    Defaults['beta'] = proba_K / np.sum(proba_K)
    Defaults['topics'] = make_bars_topics(V, K, PRNG=PRNG)

    dataset = bnpy.data.BagOfWordsData.CreateToyDataFromMixModel(**Defaults)
    dataset.to_npz('dataset.npz')
    print("Created dataset:")
    print(dataset.get_stats_summary())

    print("Counts of each true topic:")
    print(dataset.TrueParams['resp'].sum(axis=0))
