import argparse
from gensim.models import KeyedVectors


def open_model(path: str):
    w2v_model = KeyedVectors.load_word2vec_format(path, binary=True, unicode_errors='ignore')
    return w2v_model


def model_info(w2v_model: KeyedVectors):
    model_vocab = w2v_model.index_to_key
    print('Словарный запас модели состоит из {} слов'.format(len(model_vocab)))
    print('Топ 20 часто используемых слов: {}'.format(', '.join(model_vocab[:20])))


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description='information about word2vec model')
    parser.add_argument('model', type=str, help='path to model')
    args = parser.parse_args()
    # open model from file
    model = open_model(args.model)
    # print info about model
    model_info(model)
