#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from ..elsa import Preprocessing
from tqdm.auto import tqdm
from pathlib import Path
import json
import traceback
import argparse


def prepare_fasttext_xsum(preprocessor, dataset_path, save_path, split_file):
    train_val_test_split = json.load(open(split_file, 'r'))
    
    files = [x for x in Path(dataset_path).rglob('*.summary')]
    for path in tqdm(files):
        try:
            if path.stem not in train_val_test_split['train']:
                continue

            with open(path, 'r') as f:
                doc = f.read()
            text = doc.split('[SN]RESTBODY[SN]')[1]
            doc = preprocessor.preproc(text, check_stopwords=False, check_length=True, 
                                       use_lemm=False, use_stem=False, include_tf=False)
            with open(save_path, 'a') as fw:
                fw.write(f'{doc}\n')
        except KeyboardInterrupt:
            break
        except:
            print(path.stem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stopwords', nargs='*', help='path to stopwords.txt')
    parser.add_argument('-d', '--dataset', nargs='*', help='path to the dataset')
    parser.add_argument('-o', '--output', nargs='*', help='output path')
    parser.add_argument('-p', '--split', nargs='*', help='path to train-val-test split file')
    args = parser.parse_args()

    preprocessor = Preprocessing(stopwords=args.stopwords[0])
    dataset_path = args.dataset[0]
    save_path = args.output[0]
    split_file = args.split[0]

    prepare_fasttext_xsum(preprocessor, dataset_path, save_path, split_file)

