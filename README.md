<p align="center">
    <br>
    <img src="http://maksimeremeev.com/files/elsa.png" width=350/>
    <br>
<p>


# ELSA: Extractive Linking of Summarization Approaches

**Authors:** Maksim Eremeev (mae9785@nyu.edu), Mars Wei-Lun Huang (wh2103@nyu.edu), Eric Spector (ejs618@nyu.edu), Jeffrey Tumminia (jt2565@nyu.edu)

## Installation

```bash
python setup.py build
pip install .
```

## Quick Start with ELSA

```python
from elsa import Elsa

article = '''some text...
'''

abstractive_model_params = {
    'num_beams': 10,
    'max_length': 300,
    'min_length': 55,
    'no_repeat_ngram_size': 3
}

elsa = Elsa(weights=[1, 1], abstractive_base_model='bart', base_dataset='cnn', stopwords='data/stopwords.txt', 
            fasttext_model_path='datasets/cnn/elsa-fasttext-cnn.bin', 
            udpipe_model_path='data/english-ewt-ud-2.5-191206.udpipe')
            
elsa.summarize(article, **abstractive_model_params)
```

### `__init__` parameters

- `weights`: `List[float]` -- weights for TextRank and Centroid extractive summarizations.
- `abstractive_base_model`: `str` -- model used on the abstractive step. Either `'bart'` or `'pegasus'`.
- `base dataset`: `str` -- dataset used to train the abstractive model. Either `'cnn'` or `'xsum'` .
- `stopwords`: `str` -- path to the list of stopwords.
- `fasttext_model_path`: `str` -- path to the `*.bin` checkpoint of a trained FastText model (see below for the training instructions).
- `udpipe_model_path`: `str` -- path to the `*.udpipe` checkpoint of the pretrained UDPipe model (see `data` directory for the files).

### `summarize` parameters

* `factor`: `float` -- percentage (a number from 0 to 1) of sentences to keep in extractive summary (default: `0.5`)
* `use_lemm`: `bool` -- whether to use lemmatization on the preprocessing step (default: `False`)
* `use_stem`: `bool` -- whether too use stemming on the preprocessing step (default: `False`)
* `check_stopwords`: `bool` -- whether to filter stopwords on the preprocessing step (default: `True`)
* `check_length`: `bool` -- whether to filter tokens shorter than 4 symbols (default: `True`)

* `abstractive_model_params`: `dict` -- any parameters for the huggingface model's `generate` method

## Datasets used for experiments

CNN-DailyMail: [Link](https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz), original source: [Link](https://github.com/abisee/cnn-dailymail)

XSum: [Link](http://bollin.inf.ed.ac.uk/public/direct/XSUM-EMNLP18-Summary-Data-Original.tar.gz), original source: [Link](https://github.com/EdinburghNLP/XSum)

Gazeta.RU: [Link](https://www.dropbox.com/s/cmpfvzxdknkeal4/gazeta_jsonl.tar.gz), original source: [Link](https://github.com/IlyaGusev/gazeta)

#### Downloading & Extracting datasets

```bash
wget https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz
wget http://bollin.inf.ed.ac.uk/public/direct/XSUM-EMNLP18-Summary-Data-Original.tar.gz
wget https://www.dropbox.com/s/cmpfvzxdknkeal4/gazeta_jsonl.tar.gz

tar -xzf cnndm.tar.gz
tar -xzf XSUM-EMNLP18-Summary-Data-Original.tar.gz
tar -xzf gazeta_jsonl.tar.gz
```

## FastText models

#### Our trained FastText models

CNN-DailyMail: [Link](https://www.icloud.com/iclouddrive/0D92xiVCAEZa07wBde-S46r_A#elsa-fasttext-cnn)

XSum: [Link](https://www.icloud.com/iclouddrive/0bR42r-miX36v9p3rM-s3YR0Q#elsa-fasttext-gazeta)

Gazeta: [Link](https://www.icloud.com/iclouddrive/0E7muKOAdlb_EvbMPQyTN2sLw#elsa-fasttext-xsum)

See [our FastText page](https://github.com/maks5507/elsa/blob/master/fasttext_scripts/) for training details.

## UDPipe models

UDPipe models avaliable for English:

- UDPipe-English EWT: [Link](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/english-ewt-ud-2.5-191206.udpipe?sequence=17&isAllowed=y) **(Used in our experiments, see `data` directory)**
- UDPipe-English Patut: [Link](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/english-partut-ud-2.5-191206.udpipe?sequence=29&isAllowed=y)
- UDPipe-English Lines: [Link](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/english-lines-ud-2.5-191206.udpipe?sequence=30&isAllowed=y)
- UDPipe-English Gum: [Link](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/english-gum-ud-2.5-191206.udpipe?sequence=31&isAllowed=y)

Other UDPipe models: [Link](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131)

## Adaptation for Russian

As approach we use for ELSA is language-independent, we can easily adapt it to other languages. For Russian, we finetune mBart on the Gazeta dataset, train additional FastText model, and use UDPipe model built for Russian texts.

#### UDPipe models for Russian

- UDPipe-Russian Syntagrus: [Link](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/russian-syntagrus-ud-2.5-191206.udpipe?sequence=70&isAllowed=y)
- UDPipe-Russain GSD: [Link](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/russian-gsd-ud-2.5-191206.udpipe?sequence=71&isAllowed=y) **(Used in our experiments, see `data` directory)**
- UDPipe-Russian Taiga: [Link](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/russian-taiga-ud-2.5-191206.udpipe?sequence=87&isAllowed=y)

#### mBART checkpoint

HuggingFace checkpoint: [Link](https://www.icloud.com/iclouddrive/0ogqejTokfHn1tO0qiIPUldjw#mbart-checkpoint-gazeta)

## Codestyle check

Before making a commit / pull-request, please check the coding style by running the bash script in the `codestyle` directory. Make sure that your folder is included in `codestyle/pycodestyle_files.txt` list.

Your changes will not be approved if the script indicates any incongruities (this does not apply to 3rd-party code). 

Usage:

```bash
cd codestyle
sh check_code_style.sh
```

