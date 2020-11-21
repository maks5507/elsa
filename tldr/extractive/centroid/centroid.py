#
# Created by Mars Wei-Lun Huang (wh2103@nyu.edu)
#

from typing import List, Tuple, Dict
from ..util import Util


class Centroid:
    def __init__(self):
        self.util = Util()

    def summarize(self, sentences: List[str], factor: float, embedding: Dict[str, str] = None) -> List[Tuple[int, int]]:
        """
        Params:
            - factor: percent of sentences to keep in summary.
             Example: factor=0.5 restricts the summary volume to be less than 50% of the original text.
        returns: list of pairs (sentence index, score)
        """
        if not embedding or 'name' not in embedding:
            model_path = str(Path(__file__).parent.parent.joinpath('embeddings', 'model', 'cc.en.300.bin'))
            embedding = {'name': 'fasttext', 'model_path': model_path}
            sentence_vectors = self.util.build_fasttext_sentence_vectors(
                sentences, embedding.get('model_path', None))
        else:
            if embedding['name'] == 'fasttext':
                sentence_vectors = self.util.build_fasttext_sentence_vectors(
                    sentences, embedding.get('model', None))

        centroid = self.util.compute_centroid(sentence_vectors)
        scores = [self.util.compute_cosine_similarity(centroid, sentence_vector)
                  for sentence_vector in sentence_vectors]
        idx_scores = sorted(list(enumerate(scores)), key=lambda x: -x[1])
        n = int(len(sentences) * factor)
        selected_idx_scores = []
        for idx, score in idx_scores:
            if len(selected_idx_scores) == n or score < 0.5:
                break
            if len(selected_idx_scores) > 0:
                max_selected_scores = max([
                    self.util.compute_cosine_similarity(sentence_vectors[sel_idx], sentence_vectors[idx])
                    for sel_idx, _ in selected_idx_scores
                ])
                if max_selected_scores >= 0.8:
                    continue
            selected_idx_scores.append((idx, score))
        return selected_idx_scores
