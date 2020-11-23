#
# Created by Mars Wei-Lun Huang (wh2103@nyu.edu)
#

from typing import List, Tuple, Dict
from ..util import Util


class Textrank:
    def __init__(self):
        self.util = Util()

    def summarize(self, sentences: List[str], factor: float, embedding: Dict[str, str] = None) -> List[Tuple[int, int]]:
        """
        Params:
            - factor: percent of sentences to keep in summary.
            - embedding: sentence embedding for computing similarity. None is for not using embedding.
             Example: factor=0.5 restricts the summary volume to be less than 50% of the original text.
        returns: list of pairs (sentence index, score)
        """
        if not embedding or 'name' not in embedding:
            tokenized_sentences = [sentence.split() for sentence in sentences]
            sentence_vectors = self.util.build_tokenized_sentence_vectors(tokenized_sentences)
        else:
            if embedding['name'] == 'fasttext':
                sentence_vectors = self.util.build_fasttext_sentence_vectors(
                    sentences, embedding.get('model', None))

        sim_matrix = self.util.build_similarity_matrix(sentence_vectors, normalized=True)
        scores = self.util.pagerank(sim_matrix).flatten().tolist()
        index_scores = sorted(list(enumerate(scores)), key=lambda x: -x[1])
        return index_scores[:int(len(sentences) * factor)]
