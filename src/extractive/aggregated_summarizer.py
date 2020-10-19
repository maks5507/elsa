#
# Created by mae9785 (eremeev@nyu.edu)
#

from .textrank import Textrank
from .centroid import Centroid

import numpy as np


class AggregatedSummarizer:
    def __init__(self, weights: List[float]):
        self.weights = {'textrank': weights[0],
                        'centroid': weights[1]}

        self.summarizers = {'textrank': Textrank(),
                            'centroid': Centroid()}

    def summarize(self, sentences, factor):
        scores = [0] * len(sentences)

        for summarizer in self.summarizers:
            selected_sentences = self.summarizers[summarizer].summarize(sentences, factor)
            for sentence_id, score in selected_sentences:
                scores[sentence_id] += self.weights[summarizer] * score

        num_sentences = len(sentences)
        volume = np.ceil(num_sentences * factor)
        if num_sentences == 0:
            return []
        args = np.argsort(np.array(scores)).tolist()[::-1][:min(int(volume), num_sentences)]
        return args
