#
# Created by mae9785 (eremeev@nyu.edu)
#

import torch
from torch import Tensor
from torch.nn import functional as F
from typing import List


class ExtractiveAttentionMask:
    def __call__(self, mapping: List[int], sentences_scores: List[float]) -> Tensor:
        """
        return: attention mask
        """
        attention_mask = torch.zeros(len(mapping))

        for i, score in enumerate(sentences_scores):
            mask = torch.where(mapping == i, torch.tensor(1), torch.tensor(1))
            attention_mask[mask] = score

        attention_mask = F.softmax(attention_mask)

        return attention_mask




