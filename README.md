# ELSA: Extractive Linking of Summarizarion Approaches

Authors: Maksim Eremeev, Wei-Lun Huang, Eric Spector, Jeffrey Tumminia

## Codestyle check

Before making a commmit / pull-request, please check the coding style by running the bash script in the `codestyle` directory. Make sure that your folder is included in `codestyle/pycodestyle_files.txt` list.

Your changes will not be approved if the script indicates any incongruities (this does not apply to 3rd-party code). 

Usage:

```bash
cd codestyle
sh check_code_style.sh
```

## Datasets used

CNN-DailyMail: [Link](https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz), original source: [Link](https://github.com/abisee/cnn-dailymail)

XSum: [Link](http://bollin.inf.ed.ac.uk/public/direct/XSUM-EMNLP18-Summary-Data-Original.tar.gz), source: [Link](https://github.com/EdinburghNLP/XSum)

Gazeta.RU: [Link](https://www.dropbox.com/s/cmpfvzxdknkeal4/gazeta_jsonl.tar.gz), original source: [Link](https://github.com/IlyaGusev/gazeta)

#### Downloading & Extracting datasets

```shell
wget https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz
wget http://bollin.inf.ed.ac.uk/public/direct/XSUM-EMNLP18-Summary-Data-Original.tar.gz
wget https://www.dropbox.com/s/cmpfvzxdknkeal4/gazeta_jsonl.tar.gz

tar -xzf cnndm.tar.gz
tar -xzf XSUM-EMNLP18-Summary-Data-Original.tar.gz
tar -xzf gazeta_jsonl.tar.gz
```

## FastText models

#### Trained FastText models

#### Training FastText 

```shell
cd fasttext
docker build . -t elsa-fasttext-image
docker run -it -d -v "LOCAL_PATH:/root/" --name elsa-fasttext elsa-fasttext-image
fasttext 
```

## Using ELSA



