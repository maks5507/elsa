#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import torch
from torch.nn import functional as F
from typing import List


class ExtractiveAttentionMask:
    def __call__(self, mappings: List[List[int]], batch_sentences_scores: List[List[float]]) -> torch.Tensor:
        #return: attention mask
        attention_mask = torch.zeros((len(mappings), 512))

        for bid, mapping in enumerate(mappings):
            mapping_tensor = torch.Tensor(mapping)
            sentences_scores = batch_sentences_scores[bid]
            for i, score in enumerate(sentences_scores):
                mask = torch.where(mapping_tensor == i, torch.tensor(1), torch.tensor(0)).to(bool)
                attention_mask[bid][mask] = score

            mask = torch.where(mapping_tensor == -1, torch.tensor(1), torch.tensor(0)).to(bool)
            attention_mask[bid][mask] = 0.

            attention_mask[bid] = F.softmax(attention_mask[bid], dim=-1)[:512]

        return attention_mask
