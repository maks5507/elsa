# Training FastText for ELSA models

FastText is a fast char-ngram-based SGNS model. It can be rapidly trained given any set of data. We have prepared a convinient environment for the FastText training pipeline.

## Preparing the data

General recommendation for the input file:

- Do not use stemmed words
- Each document on a single line
- Do not drop stopwords
- No punctuation, lowercase

Preparing the data for summarization:

```shell
python prepare_cnn_fasttext.py -s ../data/stopwords.txt -d PATH_TO_TRAIN.TXT.SRC -o OUTPUT_TXT_PATH
python prepare_xsum_fasttext.py -s ../data/stopwords.txt -d PATH_TO_BBC-SUMMARY-DATA -o OUTPUT_TXT_PATH -s PATH_TO_THE_XSUM_TRAINING-DEV-TEST-SPLIT.json
python prepare_gazeta_fasttext.py -s ../data/stopwords.txt -d PATH_TO_GAZETA_TRAIN.JSONL -o OUTPUT_TXT_PATH
```

## Training

**Note:** You have to adjust the `lr` , `lrUpdateRate`, and `epoch` params depending on the size of your collection. 

`FASTTEXT_INPUT_FILE` is a path for the file generatied with `fasttext/prepare_{DATASET_NAME}_fasttext.py` script. `OUTPUT_NAME` is a stem of the output files `OUTPUT_NAME.bin` and `OUTPUT_NAME`.vec.

```bash
docker build . -t elsa-fasttext-image
docker run -it -d -v "HOME_DIRECTORY_ABSOLUTE_PATH:/root/" --name elsa-fasttext elsa-fasttext-image

docker attach elsa-fasttext

fasttext skipgram -input FASTTEXT_INPUT_FILE.txt -verbose 2 -minCount 5 -lr 0.05 -dim 100 -ws 5 -epoch 20 -wordNgrams 1 -minn 3 -maxn 7 -loss ns -thread 16 -neg 20 -lrUpdateRate 100 -output OUTPUT_NAME
```
