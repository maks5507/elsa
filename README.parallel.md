Usage:

```
python3 ./run_parallel.py [-j `N_JOBS`] -t `TEXT_FOLDER` -k `TEXT_PATH_MASK` -d `DATASET_NAME` -s `STOPWORDS_PATH` -f `FASTTEXT_MODEL` -u `UDPIPE_MODEL`
```

Example:

```
python3 ./run_parallel.py -j 16 -t ./cnn/ -k stories/*.story -d cnn -s ./data/stopwords.txt -f ./cnn/elsa-fasttext-cnn.bin -u ./data/english-ewt-ud-2.5-191206.udpipe
```
