from src import BartForConditionalGeneration, MBartConfig
import torch
from transformers.convert_bart_original_pytorch_checkpoint_to_pytorch import remove_ignore_keys_


def convert_fairseq_mbart_checkpoint_from_disk(checkpoint_path, hf_config_path="facebook/mbart-large-cc25"):
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    remove_ignore_keys_(state_dict)
    vocab_size = state_dict["encoder.embed_tokens.weight"].shape[0]
    mbart_config = MBartConfig.from_pretrained(hf_config_path, vocab_size=vocab_size)
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    output_projection = state_dict['decoder.output_projection.weight']
    state_dict.pop('decoder.output_projection.weight', None)
    model = BartForConditionalGeneration(mbart_config)
    model.output_projection = output_projection
    model.model.load_state_dict(state_dict)
    return model
