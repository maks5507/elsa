#
# Created by mae9785 (eremeev@nyu.edu)
#

import ufal.udpipe


class UDPipeModel:
    class Node:
        def __init__(self, token, lemmatized, pos, xpos, feats, dep_rel, anc=-1):
            self.token = token
            self.lemma = lemmatized
            self.pos = pos
            self.xpos = xpos
            self.feats = feats
            self.dep_rel = dep_rel
            self.anc = anc

    def __init__(self, path):
        self.model = ufal.udpipe.Model.load(path)

    def tokenize(self, text):
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        return self.__read(text, tokenizer)

    def read(self, text, in_format):
        input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
        return self.__read(text, input_format)

    def tag(self, sentence):
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence):
        self.model.parse(sentence, self.model.DEFAULT)

    @staticmethod
    def __read(text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        return sentences

    @staticmethod
    def write(sentences, out_format):
        output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
        output = ''
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()
        return output
