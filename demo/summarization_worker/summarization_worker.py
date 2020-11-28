#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from elsa import Elsa
from worker_compose import noexcept
import json


class SummarizationWorker:
    def __init__(self, log, list_elsa_params):
        self.log = log
        self.elsas = {}

        for elsa_id in list_elsa_params:
            elsa_params = list_elsa_params[elsa_id]
            self.elsas[elsa_id] = Elsa(**elsa_params)

    @noexcept(default_value='{"data": [], "errors": []}')
    def run(self, text, elsa_id, type):
        elsa_id = elsa_id.decode()
        text = text.decode()
        summary = self.elsas[elsa_id].summarize(text)
        self.log.info(f'Summary for input {text}: {summary}')
        return json.dumps({'data': [summary], 'errors': []})
