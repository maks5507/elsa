#
# Created by
#


import fasttext
import numpy as np


class FastTextWrapper:
    def __init__(self):
        self.model = FastText()

    def fit(self, corpora: str):
        """
        fits model given the dataset
        """
        pass

    def inference(self, sentence) -> np.ndarray:
        """
        build an embedding for given sentence
        """
        pass

    def dump(self):
        """
        saves trained model to disk
        """
        pass

    def load(self, model_path: str):
        """
        loads pretrained model from disk
        """
        self.model =
