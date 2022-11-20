import argparse
from gensim.models import KeyedVectors


def parse_args():
    parser = argparse.ArgumentParser(description='testing word2vec model')
    parser.add_argument('model', type=str, help='the filename of the model')
    parser.add_argument('--info', action='store_true', help='print info about the model')
    parser.add_argument('--similarity', type=str, help='two words to find similarity separated by a comma')
    parser.add_argument('--most_similar', type=str, help='most similar words to given key')
    parser.add_argument('--doesnt_match', type=str, help="find the word that doesn't go with the others from the "
                                                         "given list separated by a comma")
    return parser.parse_args()


def model_info(w2v_model: KeyedVectors):
    model_vocab = w2v_model.index_to_key
    print('Словарный запас модели состоит из {} слов'.format(len(model_vocab)))
    print('Топ 20 часто используемых слов: {}'.format(', '.join(model_vocab[:20])))


def similarity(w2v_model: KeyedVectors, key1: str, key2: str):
    words_similarity = w2v_model.similarity(key1, key2)
    print('Схожесть между словами "{}" и "{}" - {}'.format(key1, key2, words_similarity))


def most_similar(w2v_model: KeyedVectors, key: str):
    most_similar_words = [word[0] for word in w2v_model.most_similar(key)]
    print('Самые похожие на "{}" слова: {}'.format(key, ', '.join(most_similar_words)))


def doesnt_match(w2v_model: KeyedVectors, keys: list):
    doesnt_match_key = w2v_model.doesnt_match(keys)
    print('Слово, неподходящее к другим из списка [{}] - {}'.format(', '.join(keys), doesnt_match_key))


if __name__ == "__main__":
    # parse args
    args = parse_args()
    # open model from file
    model = KeyedVectors.load_word2vec_format(args.model, binary=True, unicode_errors='ignore')
    # if needed print info about model
    if args.info:
        model_info(model)
    # if needed print similarity between two words
    if args.similarity:
        words = args.similarity.split(',')
        similarity(model, words[0], words[1])
    if args.most_similar:
        most_similar(model, args.most_similar)
    if args.doesnt_match:
        doesnt_match(model, args.doesnt_match.split(','))
