#
# Created by
#

from typing import List, Tuple


class SentenceFiltering:
    def __init__(self):
        pass

    def filter(self, pos_dep_tags: List[List[Tuple[str, str]]]) -> List[int]:
        """
        pos_dep_tags: list of sentences, each sentence is presented by list of (POS, DEP_REL) tags of each token
        returns: indices of approved sentences
        """
        lst = list(range(len(pos_dep_tags)))
        return lst
