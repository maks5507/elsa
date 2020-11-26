#
# Created by Eric Spector
#

from typing import List
from nltk.tokenize import PunktSentenceTokenizer


class SentenceTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text: str) -> List[str]:
        """
        returns: list of sentences
        """
        sent_tokenizer = PunktSentenceTokenizer(text)
        sent_toks = sent_tokenizer.tokenize(text)
        return sent_toks
