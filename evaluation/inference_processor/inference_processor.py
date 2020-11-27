#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from elsa import Elsa
from pathlib import Path
from worker_compose import noexcept


class InferenceProcessor:
    def __init__(self, log, save_path, **elsa_params):
        self.log = log
        self.save_path = save_path
        self.elsa = Elsa(**elsa_params)

    @noexcept(default_value=None)
    def run(self, path, **abstractive_model_params):
        try:
            with open(path, 'r') as f:
                user_text = f.read()

            basename = str(Path(path).stem)
            summary = self.elsa.summarize(user_text, **abstractive_model_params)
            self.log.info(f'Summary for input {user_text}: {summary}')

            with open(f'{self.save_path}/{basename}.summary', 'w') as f:
                f.write(summary)

        except Exception:
            self.log.failure('')
