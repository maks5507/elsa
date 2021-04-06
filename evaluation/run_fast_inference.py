import os

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ec', '--extractive_config', nargs='*', help='path to extractive_config')
    parser.add_argument('-ac', '--abstractive_config', nargs='*', help='path to abstractive_config')
    parser.add_argument('-l', '--log_file', nargs='*', help='log file')
    args = parser.parse_args()

    extractive_config = args.extractive_config[0]
    abstractive_config = args.abstractive_config[0]
    log_path = args.log_file[0]
    cmd = ''
    cmd += f'python ./run.py -c {extractive_config} -l {log_path}'
    cmd += ' && '
    cmd += f'python ./run.py -c {abstractive_config} -l {log_path}'
    print(cmd)
    os.system(cmd)
