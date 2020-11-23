#
# Created by Mars Wei-Lun Huang (wh2103@nyu.edu)
#


import fasttext as ft
import numpy as np


class FastTextWrapper:
    def __init__(self):
        self.model = None

    def fit(self, corpora: str):
        """
        fits model given the dataset
        """
        pass

    def inference(self, sentence: str) -> np.ndarray:
        """
        build an embedding for given sentence
        """
        return self.model.get_sentence_vector(sentence)

    def dump(self):
        """
        saves trained model to disk
        """
        pass

    def load(self, model_path: str):
        """
        loads pretrained model from disk
        """
        self.model = ft.load_model(model_path)
