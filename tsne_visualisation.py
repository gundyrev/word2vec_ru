import argparse
from model_testing import open_model
import visualisations.similar_words as similar_words
import visualisations.vocabulary as vocabulary

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description='tsne visualization of word2vec model')
    parser.add_argument('model', type=str, help='path to model')
    parser.add_argument('type', type=int, choices=range(1, 3),
                        help='type of tsne visualisation (1 - groups of similar words, 2 - vocabulary)')
    parser.add_argument('-k', '--keys', type=str, help='words for similar words visualization, separated by comma')
    args = parser.parse_args()
    # open model from file
    model = open_model(args.model)
    # visualise
    if args.type == 1:
        similar_words.visualise(model, args.keys.split(','))
    if args.type == 2:
        vocabulary.visualise(model)
