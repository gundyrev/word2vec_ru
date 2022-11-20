import argparse
from gensim.models import KeyedVectors
import visualizations.similar_words as similar_words
import visualizations.vocabulary as vocabulary


def parse_args():
    parser = argparse.ArgumentParser(description='tsne visualizations of the word2vec model')
    parser.add_argument('model', type=str, help='the filename of the word2vec model')
    parser.add_argument('-t', '--type', type=int, default=2, choices=range(1, 3),
                        help='type of tsne visualization (1 - groups of similar words, 2 - vocabulary)')
    parser.add_argument('-k', '--keys', type=str, help='words for the similar words visualization separated by a comma')
    parser.add_argument('-s', '--save', type=str, help='filename of the image to save')
    return parser.parse_args()


if __name__ == "__main__":
    # parse args
    args = parse_args()
    # open model from file
    model = KeyedVectors.load_word2vec_format(args.model, binary=True, unicode_errors='ignore')
    # visualize
    if args.type == 1:
        similar_words.visualize(model, args.keys.split(','), args.model, args.save)
    if args.type == 2:
        vocabulary.visualize(model, args.model, args.save)
