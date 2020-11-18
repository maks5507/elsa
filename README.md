<p align="center">
    <br>
    <img src="http://maksimeremeev.com/files/elsa.png" width="300"/>
    <br>
<p>

# ELSA: Extractive Linking of Summarizarion Approaches

Authors: Maksim Eremeev, Wei-Lun Huang, Eric Spector, Jeffrey Tumminia

## Datasets

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

#### How we trained them 

**Note:** You have to adjust the `lr` , `lrUpdateRate`, and `epoch` params depending on the size of your collection. 

`FASTTEXT_INPUT_FILE` is a path for the file generatied with `fasttext/prepare_{DATASET_NAME}_fasttext.py` script. `OUTPUT_NAME` is a stem of the output files `OUTPUT_NAME.bin` and `OUTPUT_NAME`.vec.

```bash
cd fasttext

docker build . -t elsa-fasttext-image
docker run -it -d -v "HOME_DIRECTORY_ABSOLUTE_PATH:/root/" --name elsa-fasttext elsa-fasttext-image

docker attach elsa-fasttext

fasttext skipgram -input FASTTEXT_INPUT_FILE.txt -verbose 2 -minCount 5 -lr 0.05 -dim 100 -ws 5 -epoch 20 -wordNgrams 1 -minn 3 -maxn 7 -loss ns -thread 16 -neg 20 -lrUpdateRate 100 -output OUTPUT_NAME
```

## Using ELSA

```py
import elsa
```



## Codestyle check

Before making a commmit / pull-request, please check the coding style by running the bash script in the `codestyle` directory. Make sure that your folder is included in `codestyle/pycodestyle_files.txt` list.

Your changes will not be approved if the script indicates any incongruities (this does not apply to 3rd-party code). 

Usage:

```bash
cd codestyle
sh check_code_style.sh
```

