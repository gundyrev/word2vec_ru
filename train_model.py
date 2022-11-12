import argparse
import multiprocessing
from sys import argv
import pandas as pd
from gensim.models import Word2Vec
from time import time


def split_by_space(string):
    return string.split(' ')


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description='train word2vec model by prepared csv')
    parser.add_argument('path_to_csv', type=str, help='path to prepared csv file')
    parser.add_argument('path_to_model', type=str, help='path to save the trained word2vec model')
    args = parser.parse_args()
    # read csv data
    data = pd.read_csv(args.path_to_csv)['data']
    data = data.apply(split_by_space)
    # train model
    time_started = time()
    w2v_model = Word2Vec(
        min_count=5,
        window=5,
        vector_size=300,
        workers=multiprocessing.cpu_count())
    w2v_model.build_vocab(data)
    w2v_model.train(data, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    # save the model in binary format
    w2v_model.wv.save_word2vec_format(argv[2], binary=True)
    print('Модель была обучена за {} секунд и сохранена в файл {}'.format(round(time() - time_started),
                                                                          args.path_to_model))
