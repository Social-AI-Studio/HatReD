from transformers import (
    T5ForConditionalGeneration,
    EncoderDecoderModel
)

from .modeling_vl_encoder_decoder import VisionLanguageEncoderDecoderModel

def get_model(encoder_name: str, decoder_name: str, dec_tokenizer, tie_weights):
    if "t5" in encoder_name and "t5" in decoder_name:
        return T5ForConditionalGeneration.from_pretrained(decoder_name)
    else: 
        if "visualbert" in encoder_name:
            model = VisionLanguageEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_name, decoder_name, tie_weights=tie_weights)
        else:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_name, decoder_name, tie_weights=tie_weights)

        # set the decoding confgiurations
        model.config.decoder_start_token_id = dec_tokenizer.bos_token_id
        model.config.pad_token_id = dec_tokenizer.pad_token_id
        model.config.vocab_size = dec_tokenizer.vocab_size

        # make sure decoder and cross_attention is enabled
        model.is_decoder = True
        model.add_cross_attention = True

        return model


def load_model(pretrain_model_filepath):
    if "t5" in pretrain_model_filepath:
        return T5ForConditionalGeneration.from_pretrained(pretrain_model_filepath)

    if "visualbert" in pretrain_model_filepath:
        return VisionLanguageEncoderDecoderModel.from_pretrained(pretrain_model_filepath)

    return EncoderDecoderModel.from_pretrained(pretrain_model_filepath)