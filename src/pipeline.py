#
# Created by mae9785 (eremeev@nyu.edu)
#

from .preprocessing import Preprocessing, UDPipeTokenizer, SentenceTokenizer, \
    SentenceFiltering, CoreferenceResolution

from .extractive import AggregatedSummarizer
from .abstractive import mBartWrapper, AttentionMask


class Pipeline:
    def __init__(self, weights: List[float]):
        self.preprocessing = Preprocessing(stopwords='../data/stopwords.txt')
        self.udpipe_tokenizer = UDPipeTokenizer()
        self.sentence_tokenizer = SentenceTokenizer()
        self.sentence_filtering = SentenceFiltering()
        self.coreference_resolution = CoreferenceResolution()
        self.attention_mask = AttentionMask()
        self.extractive = AggregatedSummarizer(weights)
        self.abstractive = mBartWrapper()

    def summarize(self, text: str, factor: float) -> str:
        cf_text = self.coreference_resolution.resolve(text)

        sentences = self.sentence_tokenizer.tokenize(cf_text)

        preprocessed_sentences = []
        udpipe_tokens = []

        for sentence in sentences:
            preprocessed_sentences += [self.preprocessing.preproc(sentence)]
            udpipe_tokens += [self.udpipe_tokenizer.tokenize(sentence)]

        filtered_sentences = set(self.sentence_filtering.filter(udpipe_tokens))

        filtered_preprocessed_sentences = []
        for i, preprocessed_sentence in enumerate(preprocessed_sentences):
            if i in filtered_sentences:
                filtered_preprocessed_sentences += [preprocessed_sentence]

        selected_sentences = self.extractive.summarize(filtered_preprocessed_sentences, factor)
        attention_mask = self.attention_maks.generate(filtered_preprocessed_sentences, selected_sentences)
        summary = self.abstractive.summarize(filtered_preprocessed_sentences, attention_mask)

        return summary

