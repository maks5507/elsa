#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from typing import List
from .textrank import Textrank
from .centroid import Centroid
from .embeddings import FastTextWrapper


class AggregatedSummarizer:
    def __init__(self, weights: List[float], fasttext_model_path: str):
        self.weights = {'textrank': weights[0],
                        'centroid': weights[1]}

        self.summarizers = {'textrank': Textrank(),
                            'centroid': Centroid()}

        self.fasttext_model = FastTextWrapper()
        self.fasttext_model.load(fasttext_model_path)

    def summarize(self, sentences, factor):
        scores = [0] * len(sentences)

        embedding = {}
        embedding['name'] = 'fasttext'
        embedding['model'] = self.fasttext_model

        for summarizer in self.summarizers:
            selected_sentences = self.summarizers[summarizer].summarize(sentences, factor=factor,
                                                                        embedding=embedding)
            for sentence_id, score in selected_sentences:
                scores[sentence_id] += self.weights[summarizer] * score
        return scores
