from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def tsne_plot_2d(label, embeddings, image_name, words=[], a=1):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, 1))
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    plt.scatter(x, y, c=colors, alpha=a, label=label)
    for i, word in enumerate(words):
        plt.annotate(word, alpha=0.3, xy=(x[i], y[i]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom', size=10)
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('{}.png'.format(image_name), format='png', dpi=150, bbox_inches='tight')
    plt.show()


def visualise(model: KeyedVectors, image_name: str):
    words = []
    embeddings = []
    for word in list(model.index_to_key):
        embeddings.append(model[word])
        words.append(word)

    embeddings = np.array(embeddings)
    tsne_ak_2d = TSNE(n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_ak_2d = tsne_ak_2d.fit_transform(embeddings)
    tsne_plot_2d('Word2Vec Model Vocabulary', embeddings_ak_2d, image_name, a=0.1)
