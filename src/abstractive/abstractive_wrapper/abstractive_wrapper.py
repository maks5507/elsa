#
# Created by mae9785 (eremeev@nyu.edu)
#

import numpy as np
from models import *

from pretrained_models_list import pretrained


class AbstractiveWrapper:
    def __init__(self, model_name):
        pretrained_model_name = pretrained[model_name]

        self.model = pretrained[model_name][model]
        self.tokenizer = pretrained[model_name][tokenizer]


    def forward(self, sentences: List[str], attention_mask: np.ndarray, model_params: Dict):
        """
        return: summarization
        """
        text = ' '.join(sentences)

        batch = self.tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
        batch['attention_mask'] = attention_mask.unsqueeze(0)

        result = converted_model.generate(batch['input_ids'], attention_mask=batch['attention_mask'],
                                       num_beams=10, max_length=140,
                                       min_length=55, no_repeat_ngram_size=3,
                                       decoder_start_token_id=self.tokenizer.bos_token_id)

        return self.tokenizer.batch_decode(result, skip_special_tokens=True)[0]
