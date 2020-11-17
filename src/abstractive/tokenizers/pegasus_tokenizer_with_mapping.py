#
# Created by mae9785 (eremeev@nyu.edu)
#

from transformers import PegasusTokenizerFast
from .base_tokenizer_with_mapping import BaseTokenizerWithMapping


class PegasusTokenizerWithMapping(BaseTokenizerWithMapping):
    def __init__(self):
        super(PegasusTokenizerWithMapping, self).__init__(
            huggingface_tokenizer=PegasusTokenizerFast('google/pegasus-xsum'),
            truncate_left=0,
            truncate_right=1,
            starting_tokens_ids=[],
            ending_tokens_ids=[1]
        )
