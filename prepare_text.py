import argparse
import nltk
import re
import pandas as pd
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from time import time
from progress.bar import IncrementalBar

nltk.download('stopwords')

PATTERN = r"[A-Za-z0-9!#$%&'()*+,./:;<=>?@\[\]^_`{|}~—–\"]+"
STOPWORDS = stopwords.words('russian')

morph = MorphAnalyzer()

bar = IncrementalBar()


def preprocessing(line):
    # remove non-cyrillic characters
    line = re.sub(PATTERN, ' ', line)
    # convert text to lower case
    line = line.lower()
    # remove stopwords and bring words to a normalized form
    tokens = ""
    for token in line.split():
        if len(token) > 1 and token not in STOPWORDS:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            tokens = tokens + token + ' '
    bar.next()
    if len(tokens.split(' ')) > 2:
        return tokens[:-1]
    return None


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description='text preparation for further word2vec model training')
    parser.add_argument('path_to_text', type=str, help='path to text')
    parser.add_argument('path_to_csv', type=str, help='path to save the prepared csv data')
    args = parser.parse_args()
    # open text file
    time_started = time()
    data = pd.read_csv(args.path_to_text, sep='\r\n', names=['data'], engine="python")
    # remove all nones and duplicates
    data = data.dropna().drop_duplicates()
    # preprocess text
    bar = IncrementalBar('Обработка текста', max=len(data['data']))
    data = data['data'].apply(preprocessing)
    # remove all nones
    data = data.dropna()
    # save data to csv file
    data.to_csv(args.path_to_csv, index=False)
    print('\nТекст был обработан за {} секунд'.format(round(time() - time_started), 2))
