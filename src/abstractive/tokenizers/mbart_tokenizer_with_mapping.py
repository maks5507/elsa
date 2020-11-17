#
# Created by mae9785 (eremeev@nyu.edu)
#

from transformers import MBartTokenizerFast
from .base_tokenizer_with_mapping import BaseTokenizerWithMapping


class MBartTokenizerWithMapping(BaseTokenizerWithMapping):
    def __init__(self, language_id):
        super(MBartTokenizerWithMapping, self).__init__(
            huggingface_tokenizer=MBartTokenizerFast('facebook/mbart-large-cc25'),
            truncate_left=0,
            truncate_right=2,
            starting_tokens_ids=[],
            ending_tokens_ids=[2, language_id]
        )