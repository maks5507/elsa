#
# Created by
#

import re
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import pymorphy2.MorphAnalyzer
#p = Preprocessing()
#p.preproc(check_stopwords=True, use_stemming=True, use_lemmatization=False)

class Preprocessing:
    def __init__(self, stopwords: str):
        self.rgc = re.compile('[^a-zа-яё0-9-_]')
        self.tokenizer = ToktokTokenizer()

        with open(stopwords, 'r') as f:
            self.stopwords = set(f.read().split('\n'))

    def tokenize_words(self, text: str, preprocess: bool = True, **preprocess_params) -> List[str]:
        if not preprocess:
            return text.split()
        return self.preproc(text, **preprocess_params).split()

    def preproc(self, text:, check_stopwords=True) -> str:
        s = re.sub("\n", r" ", text)
        s = s.lower()
        s = self.rgc.sub(" ", s)

        final_agg = []
        tf = {}

        for i, token in enumerate(self.tokenizer.tokenize(s)):
            #check for typos etc
            if token not in self.stopwords or not check_stopwords:
                if token not in tf:
                    tf[token] = 0
                tf[token] += 1
                final_agg.append(token)
        return ' '.join(final_agg)





    def stem(self,text: str):
        pass

    def lemmatize(self,text: str):
        pass
