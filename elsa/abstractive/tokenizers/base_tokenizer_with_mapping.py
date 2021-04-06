#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import torch
import torch.nn.functional as F
from typing import Any, List, Tuple


class BaseTokenizerWithMapping:
    def __init__(self, huggingface_tokenizer: Any, truncate_left: int, truncate_right: int,
                 starting_tokens_ids: List[int], ending_tokens_ids: List[int]):
        self.main_tokenizer = huggingface_tokenizer
        self.truncate_left = truncate_left
        self.truncate_right = truncate_right
        self.starting_tokens_ids = torch.Tensor(starting_tokens_ids).to(torch.int64)
        self.ending_tokens_ids = torch.Tensor(ending_tokens_ids).to(torch.int64)

        self.bos_token_id = self.main_tokenizer.bos_token_id
        self.eos_token_id = self.main_tokenizer.eos_token_id
        self.pad_token_id = self.main_tokenizer.pad_token_id

    def tokenize(self, batch_sentences: List[List[str]]) -> Tuple[torch.Tensor, List[int]]:
        mappings = []
        tokenized_sequence_tensors = []
        for sentences in batch_sentences:
            mapping = []
            mapping += [-1] * self.starting_tokens_ids.shape[0]

            tokenized_sequence = []
            tokenized_sentences = self.main_tokenizer(sentences)['input_ids']
            for i, sentence in enumerate(tokenized_sentences):
                if isinstance(sentence, int):
                    sentence = [sentence]
                mapping += [i] * (len(sentence) - self.truncate_left - self.truncate_right)
                tokenized_sequence += sentence[self.truncate_left:-self.truncate_right]

            mapping += [-1] * self.ending_tokens_ids.shape[0]

            tokenized_sequence_tensor = torch.Tensor(tokenized_sequence).to(torch.int64)
            tokenized_sequence_tensor = torch.cat([self.starting_tokens_ids, tokenized_sequence_tensor,
                                                  self.ending_tokens_ids])
            if tokenized_sequence_tensor.shape[0] < 512:
                tokenized_sequence_tensor = F.pad(
                    input=tokenized_sequence_tensor,
                    pad=(0, 512-tokenized_sequence_tensor.shape[0]),
                    mode='constant', value=self.pad_token_id
                )
            if len(mapping) < 512:
                mapping += [-1] * (512 - len(mapping))
            tokenized_sequence_tensor = tokenized_sequence_tensor[:512].unsqueeze(0)
            tokenized_sequence_tensors.append(tokenized_sequence_tensor)
            mappings.append(mapping[:512])
        return torch.cat(tokenized_sequence_tensors, dim=0), mappings

    def decode(self, summary: torch.Tensor) -> str:
        return self.main_tokenizer.batch_decode(summary, skip_special_tokens=True)
