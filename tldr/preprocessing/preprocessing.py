#
# Created by
#

import re
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import pymorphy2.MorphAnalyzer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
import chardet


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

        words=word_tokenize(text)
        #stems=[PorterStemmer().stem(w) for w in words]
        stems = " ".join(PorterStemmer().stem(w) for word in words)
        return stems

    def lemmatize(self,text: str):

        codepage = chardet.detect(text)['encoding']
        text = text.decode(codepage)
        text = " ".join(word.lower() for word in text.split()) #lowercasing and removing short words
        text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) #deleting newlines and line-breaks
        text = re.sub('[.,:;%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) #deleting symbols
        text= " ".join(pymorphy2.MorphAnalyzer().parse(unicode(word))[0].normal_form
        for word in text.split())
        text=text.encode('utf-8')
        return text

        '''
        lemmas=Mystem().lemmatize(text)
        return ' '.join(lemmas)
        '''

        #p = Preprocessing()
        #p.preproc(check_stopwords=True, use_stemming=True, use_lemmatization=False)
