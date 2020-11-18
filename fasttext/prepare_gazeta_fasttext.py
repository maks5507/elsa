#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from ..tldr import Preprocessing
from tqdm.auto import tqdm
import json
import argparse


def prepare_fasttext_gazeta(preprocessor, dataset_path, save_path):
    with open(dataset_path, 'r') as f:
        for line in tqdm(f, total=60000):
            doc = json.loads(line)
            text = doc['text']
            preprocessed_doc = preprocessor.preproc(text, check_stopwords=False, check_length=True, 
                                                    use_lemm=True, use_stem=False, include_tf=False)
            with open(save_path, 'a') as fw:
                fw.write(f'{preprocessed_doc}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stopwords', nargs='*', help='path to stopwords.txt')
    parser.add_argument('-d', '--dataset', nargs='*', help='path to the dataset')
    parser.add_argument('-o', '--output', nargs='*', help='output path')
    args = parser.parse_args()

    preprocessor = Preprocessing(stopwords=args.stopwords[0])
    dataset_path = args.dataset[0]
    save_path = args.output[0]

    prepare_fasttext_gazeta(preprocessor, dataset_path, save_path)

