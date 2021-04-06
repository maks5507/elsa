#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from elsa import Elsa
from pathlib import Path
import os
import pickle 

from worker_compose import noexcept


class ExtractiveInferenceProcessor:
    def __init__(self, log, sentence_score_path, elsa_params):
        self.log = log
        self.sentence_score_path = sentence_score_path
        self.elsa = Elsa(**elsa_params)

    @noexcept(default_value=None)
    def run(self, path, model_params):
        with open(path, 'r') as f:
            user_text = f.read()

        sentences, sentences_scores = self.elsa.gen_sentences_scores(user_text, **model_params)
        basename = str(Path(path).stem)
        output_path = f'{self.sentence_score_path}/{basename}.pkl'
        with open(output_path, 'wb') as f_out:
            obj = {'sentences': sentences, 'sentences_scores': sentences_scores}
            pickle.dump(obj, f_out)
