#
# Created by Eric Spector
#

from typing import List, Tuple
from corpy.udpipe import Model


class UDPipeTokenizer:
    def __init__(self, udpipe_model_path):
        self.udpipe_model = Model(udpipe_model_path)

    def tokenize(self, sentence: str) -> List[Tuple[str, str]]:
        """
        return: list of pairs of tags (POS, DEP_REL) for each token in the sentence
        """
        s = list(self.udpipe_model.process(sentence))
        lst = [(item.upostag, item.deprel) for item in s[0].words if item.upostag != '<root>']
        return lst
