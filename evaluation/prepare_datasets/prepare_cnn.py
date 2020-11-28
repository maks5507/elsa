#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import argparse
import re


def prepare_evaluation_cnn_dataset(val_src_path, val_tgt_path, src_path, tgt_path):
    with open(val_src_path, 'r') as f:
        for i, line in enumerate(f):
            with open(f'{src_path}/{i}.src', 'w') as ff:
                ff.write(line)

    with open(val_tgt_path, 'r') as f:
        for i, line in enumerate(f):
            line = re.sub('<t>', '', line)
            line = re.sub('</t>', '', line)
            line = re.sub(' +', ' ', line)
            line = re.sub(' . ', '. ', line)
            line = re.sub(' , ', ', ', line)
            with open(f'{tgt_path}/{i}.summary', 'w') as ff:
                ff.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-vs', '--val_source_path', nargs='*', help='path to val.src')
    parser.add_argument('-vt', '--val_target_path', nargs='*', help='path to val.tgt')
    parser.add_argument('-s', '--src_output_path', nargs='*', help='source output pathh')
    parser.add_argument('-t', '--tgt_output_path', nargs='*', help='target output path')
    args = parser.parse_args()

    prepare_evaluation_cnn_dataset(args.val_source_path[0], args.val_target_path[0],
                                   args.src_output_path[0], args.tgt_output_path[0])
