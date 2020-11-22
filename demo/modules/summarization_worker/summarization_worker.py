#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from elsa import Elsa
from .. import noexcept


class SummarizationWorker:
    def __init__(self, log, **list_elsa_params):
        self.log = log
        self.elsas = []

        for elsa_id in list_elsa_params:
            elsa_params = list_elsa_params[elsa_id]
            self.elsas[elsa_id] = Elsa(**elsa_params)

    @noexcept(default_value={"data": [], "errors": []})
    def run(self, user_text, elsa_id):
        summary = self.elsas[elsa_id].summarize(user_text)
        self.log.info(f'Summary for input {user_text}: {summary}')
        return {'data': [summary], 'errors': []}
