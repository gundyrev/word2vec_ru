import argparse
import multiprocessing
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from time import time
from progress.bar import IncrementalBar

bar = IncrementalBar()


class Callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        bar.next()
        self.epoch += 1


def parse_args():
    parser = argparse.ArgumentParser(description='train word2vec model by the prepared csv data')
    parser.add_argument('csv_filename', type=str, help='the filename of the prepared csv data')
    parser.add_argument('model_filename', type=str, help='the filename to save the trained word2vec model')
    parser.add_argument('--epochs', type=int, default=30, help='training parameter - number of epochs')
    parser.add_argument('--min_count', type=int, default=5, help='training parameter - minimum word frequency')
    parser.add_argument('--window', type=int, default=5, help='training parameter - window')
    parser.add_argument('--vector_size', type=int, default=300, help='training parameter - vector size')
    parser.add_argument('--negative', type=int, default=20, help='training parameter - negative sampling')
    return parser.parse_args()


def split_by_space(string):
    return string.split(' ')


if __name__ == "__main__":
    # parse args
    args = parse_args()
    # read csv data
    data = pd.read_csv(args.csv_filename)['data']
    data = data.apply(split_by_space)
    # train model
    time_started = time()
    w2v_model = Word2Vec(
        min_count=args.min_count,
        window=args.window,
        vector_size=args.vector_size,
        negative=args.negative,
        workers=multiprocessing.cpu_count())
    w2v_model.build_vocab(data)
    bar = IncrementalBar('Обучение модели', max=args.epochs)
    w2v_model.train(data, total_examples=w2v_model.corpus_count, epochs=args.epochs, report_delay=1,
                    callbacks=[Callback()])
    # save the model in binary format
    w2v_model.wv.save_word2vec_format(args.model_filename, binary=True)
    print(f'\nМодель была обучена за {int(time() - time_started)} секунд')
