#
# Created by
#
from nltk.tokenize import word_tokenize,
from nltk.tokenize import sent_tokenize

class SentenceFiltering:
    def __init__(self):
        pass

    def filter(self, pos_dep_tags: List[List[Tuple(str, str)]]) -> List[int]:
        """
        pos_dep_tags: list of sentences, each sentence is presented by list of (POS, DEP_REL) tags of each token
        returns: indices of approved sentences
        """
        lst=[]
        for i, (p,d) in enumerate(pos_dep_tags):
            if model.tokenize() == p,d:
                lst.append(i)
        return lst
