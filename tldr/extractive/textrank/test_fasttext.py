from pathlib import Path
from .textrank import Textrank

textrank = Textrank()

test_data_path = './data/cnn_story_tokenized_test'
stopwords_path = './data/stopwords.txt'

with open(stopwords_path) as fp:
    stopwords = [line.strip() for line in fp.readlines()]
    stopword_set = set(stopwords)

sentences = []
filtered_sentences = []
with open(test_data_path) as fp:
    for line in fp.readlines():
        line = line.strip()
        if len(line) == 0:
            continue
        tokens = [token for token in line.strip().split()
                  if token not in stopword_set
                  and not (token[0] == '-' and token[-1] == '-')]
        sentences.append(line)
        filtered_sentences.append(' '.join(tokens))

factor = 0.5
model_path = str(Path(__file__).parent.parent.
                 joinpath('embeddings', 'model', 'cc.en.300.bin'))
embedding = {'name': 'fasttext', 'model_path': model_path}
sentence_index_scores = textrank.summarize(filtered_sentences, factor, embedding)
print(sentence_index_scores)
print([(sentences[idx], score) for idx, score in sentence_index_scores])
