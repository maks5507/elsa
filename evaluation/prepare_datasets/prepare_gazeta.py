#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import argparse
import json


def prepare_evaluation_cnn_dataset(val_json_path, src_path, tgt_path):
    with open(val_json_path, 'r') as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            with open(f'{src_path}/{i}.src', 'w') as ff:
                ff.write(doc['text'])
            with open(f'{tgt_path}/{i}.summary', 'w') as ff:
                ff.write(doc['summary'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--val_jsonl_path', nargs='*', help='path to val.jsonl')
    parser.add_argument('-s', '--src_output_path', nargs='*', help='source output pathh')
    parser.add_argument('-t', '--tgt_output_path', nargs='*', help='target output path')
    args = parser.parse_args()

    prepare_evaluation_cnn_dataset(args.val_jsonl_path[0], args.src_output_path[0], args.tgt_output_path[0])
