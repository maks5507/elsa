#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from elsa import Elsa
from pathlib import Path
import os

from worker_compose import noexcept


class InferenceProcessor:
    def __init__(self, log, save_path, elsa_params):
        self.log = log
        self.save_path = save_path
        self.elsa = Elsa(**elsa_params)

    @noexcept(default_value=None)
    def run(self, path, model_params):
        with open(path, 'r') as f:
            user_text = f.read()

        self.log.info(f'{path}')
        basename = str(Path(path).stem)

        output_path = f'{self.save_path}/{basename}.summary'

        if os.path.exists(output_path):
            return

        summary = self.elsa.summarize(user_text, **model_params)
        self.log.info(f'Summary for input {basename}: {summary}')

        with open(output_path, 'w') as f:
            f.write(summary)

