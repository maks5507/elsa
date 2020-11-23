#!/usr/bin/python

import argparse
import os
import json

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--texts", type=str, required=True,
                        help="text folder")
    parser.add_argument("-k", "--mask", type=str, required=True,
                        help="text path pattern")
    parser.add_argument("-w", "--extractive_weight", type=float,
                        default=1, help="extractive model weight")
    parser.add_argument("-W", "--abstractive_weight", type=float,
                        default=1, help="abstractive model weight")
    parser.add_argument("-b", "--base_model", type=str, default="bart",
                        help="abstractive base model")
    parser.add_argument("-d", "--base_dataset", type=str, required=True,
                        help="base dataset")
    parser.add_argument("-s", "--stopwords", type=str, required=True,
                        help="stopwords path")
    parser.add_argument("-f", "--fasttext_model", type=str, required=True,
                        help="fasttext model path")
    parser.add_argument("-u", "--udpipe_model", type=str, required=True,
                        help="udpipe_model_path")
    parser.add_argument("-B", "--beams", type=int, default=10,
                        help="number of beams")
    parser.add_argument("-M", "--max_len", type=int, default=300,
                        help="maximal sentence length")
    parser.add_argument("-m", "--min_len", type=int, default=55,
                        help="minimal sentence length")
    parser.add_argument("-r", "--no_repeat_ngram", type=int, default=3,
                        help="non-repeated ngram size")
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="number of jobs")
    return parser


def gen_config(args, config_path):
    config = {
        "summarization_worker": {
            "mask": args.mask, "texts": args.texts,
            "prefix": "elsa.parallel.modules",
            "name": "summarization_worker",

            "init_args": {
                "weights": [args.extractive_weight, args.abstractive_weight],
                "abstractive_base_model": args.base_model,
                "base_dataset": args.base_dataset, "stopwords": args.stopwords,
                "fasttext_model_path": args.fasttext_model, 
                "udpipe_model_path": args.udpipe_model
            },
     
            "run_args": {
                "num_beams": args.beams,
                "max_length": args.max_len, "min_length": args.min_len,
                "no_repeat_ngram_size": args.no_repeat_ngram
            },

            "n_jobs": args.jobs,
            "add_process_num": False,

            "depends_on": [], 
            "mode": "processor"
        }
    }
    json.dump(config, open(config_path, "w"))

def run_parallel(config_path, log_path):
    exec_path = "elsa.parallel.worker.run"
    cmd = "OMP_NUM_THREADS=1 " \
          "NEURALCOREF_CACHE=/tmp2/r02922041/.cache/neuralcoref " \
          "TRANSFORMERS_CACHE=/tmp2/r02922041/.cache/transformer " \
          "python3 -m {exec_path} -c {cfg_path} -l {log_path}"
    cmd = cmd.format(exec_path=exec_path, cfg_path=config_path,
                     log_path=log_path)
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    config_path = "./elsa/parallel/modules/summarization_worker/config.json"
    log_path = "./elsa/parallel/modules/summarization_worker/log.txt"
    gen_config(args, config_path)
    run_parallel(config_path, log_path)
