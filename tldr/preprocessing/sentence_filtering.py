#
# Created by
#
from nltk.tokenize import word_tokenize,
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import heapq

class SentenceFiltering:
    def __init__(self):
        pass

    def filter(self, pos_dep_tags: List[List[Tuple(str, str)]]) -> List[int]:
        """
        pos_dep_tags: list of sentences, each sentence is presented by list of (POS, DEP_REL) tags of each token
        returns: indices of approved sentences
        """
        '''
        lst=[]
        for i,sentence in enumerate(pos_dep_tags):
            include=1
            for (word,pos) in sentence:
                if pos=='PRP':
                    include=0
                    break

            if (include):
                lst.append(i)

        return lst
        '''
        stopwords = (nltk.corpus.stopwords.words('english')+
        nltk.corpus.stopwords.words('russian'))

        word_frequencies = {}
        for sentence in pos_dep_tags:
            for word in sentence:
                if word not in stopwords:
                    if word not in word_frequencies.keys():
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1

        maximum_frequency = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

        sentence_scores = {}
        for sent in sentence_list:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 30: #Max words in sentence
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]

        summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)
