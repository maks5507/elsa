#
# Created by Eric Spector
#

import neuralcoref
import spacy


class CoreferenceResolution:
    def __init__(self, spacy_model: str = 'en_core_web_sm', greedyness: float = 0.45):
        self.nlp = spacy.load(spacy_model)
        neuralcoref.add_to_pipe(self.nlp, greedyness=greedyness)

    def resolve(self, text: str) -> str:
        """
        return: text with resolved coreferences
        """
        res = self.nlp(text)._.coref_resolved
        return res
