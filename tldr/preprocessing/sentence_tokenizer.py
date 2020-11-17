#
# Created by
#
from nltk.tokenize import word_tokenize
from nltk.tokenize import PunktSentenceTokenizer

class SentenceTokenizer:
    def __init__(self):
        self.sent_toks=[]

    def tokenize(self, text: str) -> List[str]:
        """
        returns: list of sentences
        """
        sent_tokenizer = PunktSentenceTokenizer(text)
        self.sent_toks = sent_tokenizer.tokenize(text)
        return self.sent_toks


#a=SentenceTokenizer
#print(a.tokenize('whats up? whats new?'))
