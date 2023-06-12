from json import encoder
import torch
import numpy as np
import random

import config
import os


from models import get_model, load_model
from builder import get_datasets
from trainer import get_trainer
# from trainer.common import GenerationResults

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer

from transformers import get_linear_schedule_with_warmup, EncoderDecoderModel
from utils import Logger, feature_collate_fn, caption_collate_fn
# from models.temp import EncoderDecoderModel

from models.modeling_vl_encoder_decoder import VisionLanguageEncoderDecoderModel

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_tokenizers(encoder_name, decoder_name):

    # temp fix for visualbert
    if 'visualbert' in encoder_name:
        encoder_name = 'bert-base-uncased'

    enc_tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=False)
    if enc_tokenizer.pad_token is None:
        enc_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    logger.write(f"Encoder Tokenizer initialised from {encoder_name}")

    dec_tokenizer = AutoTokenizer.from_pretrained(decoder_name, use_fast=False)
    if dec_tokenizer.pad_token is None:
        dec_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    logger.write(f"Decoder Tokenizer initialised from {decoder_name}")

    return enc_tokenizer, dec_tokenizer

def get_sep_token(encoder_name):
    if 't5' in encoder_name:
        return "</s>"
    elif 'roberta' in encoder_name:
        return "</s></s>"
    elif 'visualbert' in encoder_name:
        return "[SEP]"
    else:
        raise ValueError(f"the separator token for '{encoder_name}' is not implemented...")


def get_model_params(model, fix_layers, weight_decay):
    params = {}
    for n, p in model.named_parameters():
        if fix_layers > 0:
            if 'encoder.layer' in n:
                try:
                    layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                except:
                    print(n)
                    raise Exception("")
                if layer_num >= fix_layers:
                    print('yes', n)
                    params[n] = p
                else:
                    print('no ', n)
            elif 'embeddings' in n:
                print('no ', n)
            else:
                print('yes', n)
                params[n] = p
        else:
            params[n] = p

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    return optimizer_grouped_parameters

if __name__=='__main__':
    opt = config.parse_opt("train")

    # Set seed
    device = torch.device(opt.CUDA_DEVICE)
    set_seed(opt.SEED)

    # Define logger and model save path
    log_path = os.path.join(opt.DATASET, f"{opt.SAVE_FILEPATH}.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Create logger for writing logs
    logger = Logger(log_path)
    logger.log_hyperpara(opt)
    
    # Create model configurations
    enc_tokenizer, dec_tokenizer = get_tokenizers(opt.ENC_MODEL_NAME, opt.DEC_MODEL_NAME)

    # Generate datasets
    sep_token = get_sep_token(opt.ENC_MODEL_NAME)
    train_set, test_set = get_datasets(
        opt.TRAIN_ANNOTATIONS,
        opt.TEST_ANNOTATIONS,
        opt.USE_ENTITIES,
        opt.USE_DEMOGRAPHICS,
        opt.CLEAN_CONTRACTIONS,
        opt.DEC_MODEL_NAME,
        sep_token,
        captions_fp=opt.CAPTIONS_FILEPATH,
        features_fp=opt.FEATURES_FILEPATH
    )

    if opt.CAPTIONS_FILEPATH:
        collate_fn = caption_collate_fn
        # from trainer.decoder import train_fn, test_fn
    if opt.FEATURES_FILEPATH:
        # from trainer.vision_language import train_fn, test_fn
        collate_fn = feature_collate_fn

    train_loader = DataLoader(train_set,
                            opt.BATCH_SIZE,
                            shuffle=True,
                            num_workers=1,
                            collate_fn=collate_fn)
                            
    test_loader = DataLoader(test_set,
                           opt.BATCH_SIZE,
                           shuffle=False,
                           num_workers=1,
                           collate_fn=collate_fn)

    logger.write(f"dataset: {type(train_set)}")
    logger.write(f"encoder tokenizer: {type(enc_tokenizer)}")
    logger.write(f"decoder tokenizer: {type(dec_tokenizer)}")
    logger.write(f"An example from train_set: {train_set[0]}")

    logger.write('Length of training set: %d, length of testing set: %d' %
                 (len(train_loader.dataset),len(test_loader.dataset)))

    # Construct models
    model = get_model(opt.ENC_MODEL_NAME, opt.DEC_MODEL_NAME, dec_tokenizer, opt.TIE_WEIGHTS)

    if isinstance(model, EncoderDecoderModel) or isinstance(model, VisionLanguageEncoderDecoderModel):
        # set the decoding confgiurations
        model.config.decoder_start_token_id = dec_tokenizer.bos_token_id
        model.config.pad_token_id = dec_tokenizer.pad_token_id
        model.config.vocab_size = dec_tokenizer.vocab_size

        # make sure decoder and cross_attention is enabled
        model.is_decoder = True
        model.add_cross_attention = True

        # resize token embeddings
        model.decoder.resize_token_embeddings(len(dec_tokenizer))
    else:
        model.resize_token_embeddings(len(dec_tokenizer))

    model = model.to(device)
    model_params = get_model_params(model, opt.FIX_LAYERS, opt.WEIGHT_DECAY)

    # Determine optimizer, criterion and scheduler
    optim = AdamW(model_params, lr=opt.LR_RATE, eps=opt.EPS)
    num_training_steps=len(train_loader) * opt.EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=opt.WARM_UP, num_training_steps=num_training_steps
    )

    model_filename = os.path.join(opt.DATASET, f"{opt.SAVE_FILEPATH}")
    logger.write(f"{type(model)} {opt.ENC_MODEL_NAME} {opt.DEC_MODEL_NAME}")

    trainer = get_trainer(
        model=model, model_filename=model_filename, 
        enc_tokenizer=enc_tokenizer, dec_tokenizer=dec_tokenizer,
        train_loader=train_loader, valid_loader=test_loader, test_loader=test_loader,
        optim=optim, scheduler=scheduler, sep_token=sep_token, target_mode=opt.TARGET_MODE,
        gradient_accumulation_steps=opt.GRADIENT_ACCUMULATION_STEPS, device=device,
        epoches=opt.EPOCHS, logger=logger
    )

    if opt.TRAIN:
        trainer.train()
    else:
        # Load test model 
        model_filepath = os.path.join(opt.TEST_MODEL_FILEPATH)
        trainer.model = load_model(model_filepath).to(device)

        # test model
        trainer.test()