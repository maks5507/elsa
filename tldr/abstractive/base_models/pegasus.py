#
# Created by Google and HuggingFace
# Adapted by Maksim Eremeev (mae9785@nyu.edu)
#


from transformers.configuration_pegasus import PegasusConfig
from . import BartForConditionalGeneration


class PegasusForConditionalGeneration(BartForConditionalGeneration):
    config_class = PegasusConfig
    authorized_missing_keys = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        "model.encoder.embed_positions",
        "model.decoder.embed_positions",
    ]
    keys_to_never_save = [
        "model.encoder.embed_positions.weight",
        "model.decoder.embed_positions.weight",
    ]
