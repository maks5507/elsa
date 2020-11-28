#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from pathlib import Path
import argparse
import json
import traceback


def prepare_evaluation_xsum_dataset(dataset_path, split_file, src_path, tgt_path):
    train_val_test_split = json.load(open(split_file, 'r'))

    files = [x for x in Path(dataset_path).rglob('*.summary')]
    for path in files:
        try:
            if path.stem not in train_val_test_split['test']:
                continue

            with open(path, 'r') as f:
                doc = f.read()

            text = doc.split('[SN]RESTBODY[SN]')
            source = text[1]
            text = text[0]

            text = text.split('[SN]FIRST-SENTENCE[SN]')[1].split('\n')[1]

            with open(f'{src_path}/{path.stem}.src', 'w') as fw:
                fw.write(f'{source}')

            with open(f'{tgt_path}/{path.stem}.summary', 'w') as fw:
                fw.write(f'{text}')
        except KeyboardInterrupt:
            break
        except:
            print(traceback.format_exc())
            print(path.stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', nargs='*', help='path to dataset folder')
    parser.add_argument('-sf', '--split_file', nargs='*', help='path to split_file')
    parser.add_argument('-s', '--src_output_path', nargs='*', help='source output pathh')
    parser.add_argument('-t', '--tgt_output_path', nargs='*', help='target output path')
    args = parser.parse_args()

    prepare_evaluation_xsum_dataset(args.dataset_path[0], args.split_file[0],
                                    args.src_output_path[0], args.tgt_output_path[0])
