#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from typing import List

from ..extractive_attention_mask import ExtractiveAttentionMask
from ..tokenizers import BartTokenizerWithMapping, PegasusTokenizerWithMapping, MBartTokenizerWithMapping
from ..base_models import BartForConditionalGeneration, PegasusForConditionalGeneration


class AbstractiveModel:
    datasets_mapping = {
        'bart': {
            'cnn': 'facebook/bart-large-cnn',
            'xsum': 'facebook/bart-large-xsum'
        },
        'mbart': {
            'gazeta.ru': '../../../data/mbart-checkpoint-gazeta.pt'
        },
        'pegasus': {
            'cnn': 'google/pegasus-cnn',
            'xsum': 'google/pegasus-xsum',
            'gigaword': 'google/pegasus-gigaword'
        }
    }

    setup_mapping = {
        'bart': {
            'base_model': BartForConditionalGeneration,
            'tokenizer': BartTokenizerWithMapping
        },
        'mbart': {
            'base_model': BartForConditionalGeneration,
            'tokenizer': MBartTokenizerWithMapping
        },
        'pegasus': {
            'base_model': PegasusForConditionalGeneration,
            'tokenizer': PegasusTokenizerWithMapping
        }
    }

    def __init__(self, base_model_name, dataset):
        self.base_model_name = base_model_name.lower()
        self.dataset = dataset.lower()

        self.tokenizer = self.setup_mapping[self.base_model_name]['tokenizer']()

        self.base_model_class = self.setup_mapping[self.base_model_name]['base_model']
        self.base_model = self.base_model_class.from_pretrained(self.datasets_mapping[self.base_model_name][self.dataset])

        self.extractive_attention_mask = ExtractiveAttentionMask()

    def __call__(self, sentences: List[str], sentence_scores: List[int], **base_model_params):
        """
        return: summary
        """
        tokenized_sequence, mapping = self.tokenizer.tokenize(sentences)
        attention_mask = self.extractive_attention_mask(mapping, sentence_scores)

        summary = self.base_model.generate(input_ids=tokenized_sequence, attention_mask=attention_mask,
                                           decoder_start_token_id=self.tokenizer.bos_token_id, **base_model_params)

        return self.tokenizer.decode(summary)
