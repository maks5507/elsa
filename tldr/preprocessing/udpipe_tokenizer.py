#
# Created by
#

from .udpipe_wrapper import UDPipeModel
from corpy.udpipe import Model

class UDPipeTokenizer:
    def __init__(self):
        pass

    def tokenize(self, sentence: str): -> List[Tuple(str, str)]:
        """
        return: list of pairs of tags (POS, DEP_REL) for each token in the sentence
        """
        m=Model(sentence)
        s=list(m.process())
        lst=[(item.upostag,item.deprel) for item in s]
        return lst
