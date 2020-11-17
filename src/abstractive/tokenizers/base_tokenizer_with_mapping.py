#
# Created by mae9785 (eremeev@nyu.edu)
#

import torch
from typing import Any, List


class BaseTokenizerWithMapping:
    def __init__(self, huggingface_tokenizer: Any, truncate_left: int, truncate_right: int,
                 starting_tokens_ids: List[int], ending_tokens_ids: List[int]):
        self.main_tokenizer = huggingface_tokenizer
        self.truncate_left = truncate_left
        self.truncate_right = truncate_right
        self.starting_tokens_ids = torch.Tensor(starting_tokens_ids).to(torch.int64)
        self.ending_token_ids = torch.Tensor(ending_tokens_ids).to(torch.int64)

    def tokenize(self, sentences):
        mapping = []
        tokenized_sequence = []

        tokenized_sentences = self.main_tokenizer(sentences)['input_ids']
        for i, sentence in tokenized_sentences:
            mapping += [i] * (len(sentence) - self.truncate_left - self.truncate_right)
            tokenized_sentences += [sentence[self.truncate_left:-self.truncate_right]]

        tokenized_sequence_tensor = torch.Tensor(tokenized_sequence).to(torch.int64)
        tokenized_sequence_tensor = torch.cat([self.starting_tokens_ids, tokenized_sequence_tensor,
                                              self.ending_token_ids])

        return tokenized_sequence_tensor, mapping
