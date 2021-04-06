#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#
import torch

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
            'gazeta': '/root/tldr-project/data/mbart-pretrained-gazeta'
        },
        'pegasus': {
            'cnn': 'google/pegasus-cnn_dailymail',
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
        self.base_model = self.base_model_class.from_pretrained(
            self.datasets_mapping[self.base_model_name][self.dataset]
        )

        self.extractive_attention_mask = ExtractiveAttentionMask()

    def __call__(self, batch_sentences: List[List[str]],
                 batch_sentence_scores: List[List[int]],
                 use_gpu = False, **base_model_params):
        """
        return: summary
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
                 if use_gpu else 'cpu'

        tokenized_sequence, mappings = self.tokenizer.tokenize(batch_sentences)
        attention_mask = self.extractive_attention_mask(mappings, batch_sentence_scores)
        tokenized_sequence = tokenized_sequence.to(device)
        attention_mask = attention_mask.to(device)
        self.base_model = self.base_model.to(device)
        batch_summary = self.base_model.generate(input_ids=tokenized_sequence, attention_mask=attention_mask,
                                                 decoder_start_token_id=self.tokenizer.bos_token_id, **base_model_params)
        return self.tokenizer.decode(batch_summary)
