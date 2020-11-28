#
# Created by Eric Spector
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
        filtered_sentences = []

        for i, sentence_tags in enumerate(pos_dep_tags):
            num_sbj = 0
            num_sbj_prp = 0

            num_verbs = 0
            for tag, dep in sentence_tags:
                if dep == 'nsubj':
                    num_sbj += 1
                if dep == 'nsubj' and tag in ['PRON', 'DET', 'PRP$', 'PRP', 'DT']:
                    num_sbj_prp += 1
                if 'VB' in tag or 'VERB' in tag:
                    num_verbs += 1

            if num_sbj_prp > 0:
                continue
            if num_verbs == 0:
                continue

            filtered_sentences += [i]
        return filtered_sentences
