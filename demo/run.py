#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import pathmagic
pathmagic.add_to_path(1)

from worker_compose import Launcher
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='*', help='path to config')
    parser.add_argument('-l', '--log_file', nargs='*', help='log file')
    args = parser.parse_args()

    with open(args.config[0], 'r') as f:
        config = f.read()

    launcher = Launcher(log_file=args.log_file[0])

    launcher.launch(config)
