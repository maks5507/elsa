#
# Created by
#

from .udpipe_wrapper import UDPipeModel


class UDPipeTokenizer:
    def __init__(self):
        pass

    def tokenize(self, sentence: str) -> List[Tuple(str, str)]:
        """
        return: list of pairs of tags (POS, DEP_REL) for each token in the sentence
        """
        pass
