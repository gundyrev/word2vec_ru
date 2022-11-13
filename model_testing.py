import argparse
from gensim.models import KeyedVectors


def parse_args():
    parser = argparse.ArgumentParser(description='testing word2vec model')
    parser.add_argument('model', type=str, help='the filename of the model')
    parser.add_argument('--info', action='store_true', help='print info about the model')
    parser.add_argument('--similarity', type=str, help='two words to find similarity separated by a comma')
    return parser.parse_args()


def open_model(path: str):
    w2v_model = KeyedVectors.load_word2vec_format(path, binary=True, unicode_errors='ignore')
    return w2v_model


def model_info(w2v_model: KeyedVectors):
    model_vocab = w2v_model.index_to_key
    print('Словарный запас модели состоит из {} слов'.format(len(model_vocab)))
    print('Топ 20 часто используемых слов: {}'.format(', '.join(model_vocab[:20])))


def similarity(w2v_model: KeyedVectors, word1: str, word2: str):
    words_similarity = w2v_model.similarity(word1, word2)
    print('Схожесть между словами "{}" и "{}" - {}'.format(word1, word2, words_similarity))


if __name__ == "__main__":
    # parse args
    args = parse_args()
    # open model from file
    model = open_model(args.model)
    # if needed print info about model
    if args.info:
        model_info(model)
    # if needed print similarity between two words
    if args.similarity:
        words = args.similarity.split(',')
        similarity(model, words[0], words[1])
