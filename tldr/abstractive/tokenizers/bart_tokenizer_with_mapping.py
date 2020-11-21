#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from transformers import BartTokenizerFast
from .base_tokenizer_with_mapping import BaseTokenizerWithMapping


class BartTokenizerWithMapping(BaseTokenizerWithMapping):
    def __init__(self):
        super(BartTokenizerWithMapping, self).__init__(
            huggingface_tokenizer=BartTokenizerFast.from_pretrained('facebook/bart-large-cnn'),
            truncate_left=1,
            truncate_right=1,
            starting_tokens_ids=[0],
            ending_tokens_ids=[2]
        )
