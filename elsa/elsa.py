#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from typing import List

from .preprocessing import Preprocessing, UDPipeTokenizer, SentenceTokenizer, \
    SentenceFiltering, CoreferenceResolution

from .extractive import AggregatedSummarizer
from .abstractive import AbstractiveModel


class Elsa:
    def __init__(self, weights: List[float] = (1, 1), abstractive_base_model: str = 'bart', base_dataset: str = 'cnn',
                 stopwords: str = '../data/stopwords.txt', fasttext_model_path: str = 'datasets/cnn/elsa-fasttext-cnn.bin',
                 udpipe_model_path: str = '../data/eng.udpipe'):
        self.preprocessing = Preprocessing(stopwords=stopwords)
        self.udpipe_tokenizer = UDPipeTokenizer(udpipe_model_path)
        self.sentence_tokenizer = SentenceTokenizer()
        self.sentence_filtering = SentenceFiltering()
        self.coreference_resolution = CoreferenceResolution()
        self.extractive = AggregatedSummarizer(weights, fasttext_model_path)
        self.abstractive_model = AbstractiveModel(abstractive_base_model, base_dataset)

    def summarize(self, text: str, factor: float = 0.5, **abstractive_params) -> str:
        cf_text = self.coreference_resolution.resolve(text)

        sentences = self.sentence_tokenizer.tokenize(cf_text)

        preprocessed_sentences = []
        udpipe_tokens = []

        for sentence in sentences:
            preprocessed_sentences += [self.preprocessing.preproc(sentence)]
            udpipe_tokens += [self.udpipe_tokenizer.tokenize(sentence)]

        filtered_sentences = self.sentence_filtering.filter(udpipe_tokens)
        filtered_sentences_set = set(filtered_sentences)

        filtered_preprocessed_sentences = []
        for i, preprocessed_sentence in enumerate(preprocessed_sentences):
            if i in filtered_sentences_set:
                filtered_preprocessed_sentences += [preprocessed_sentence]

        filtered_sentences_scores = self.extractive.summarize(filtered_preprocessed_sentences, factor=factor)
        sentences_scores, cur_pointer = [], 0
        for i in range(len(sentences)):
            if i in filtered_sentences_set:
                sentences_scores += [filtered_sentences_scores[cur_pointer]]
                cur_pointer += 1
            else:
                sentences_scores += [0]

        summary = self.abstractive_model(sentences, sentences_scores, **abstractive_params)
        return summary

