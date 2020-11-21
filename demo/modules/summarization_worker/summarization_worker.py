#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import traceback

from .. import Elsa
from .. import noexcept


class SummarizationWorker:
    def __init__(self, log, **elsa_params):
        self.log = log
        self.elsa = Elsa(**elsa_params)

    @noexcept(default_value={"data": [], "errors": []})
    def run(self, user_text):
        try:
            summary = self.elsa.summarize(user_text)
            self.log.info(f'Summary for input {user_text}: {summary}')

            return {'data': [summary], 'errors': []}
        except Exception:
            self.log.failure('')
            return {'data': [], 'errors': [traceback.format_exc()]}