import os

from typing import List
from .utils import Evaluator, SaveCheckpoint


class BaseTrainer:
    def __init__(self, model, model_filename, enc_tokenizer, dec_tokenizer,
                 train_loader, valid_loader, test_loader,
                 optim, scheduler, sep_token, target_mode,
                 gradient_accumulation_steps, device, epoches, logger):

        self.evaluator = Evaluator()
        self.checkpoint = SaveCheckpoint(model, model_filename, logger)

        self.model = model
        self.model_filename = model_filename
        self.logger = logger

        # optimizers, schedulers and gradients accumulation
        self.optim = optim
        self.scheduler = scheduler
        self.sep_token = sep_token
        self.target_mode = target_mode
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device

        # tokenizers
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer

        # data loaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.epoches = epoches

        self.decoding_strategies = [
            ("greedy", {"do_sample": False, "max_length": 50, "num_beams": 1}),  # Greedy Decoding
            ("beam", {"do_sample": False, "max_length": 50, "num_beams": 5}),  # Beam Search = 5
            ("top-p", {
                "do_sample": True,
                "max_length": 50,
                "top_p": 0.92,
                "top_k": 0
            }) # Top-P Sampling
        ]

    def _process_inputs(self, input_sents: List[str]):

        tokenized_sents = self.enc_tokenizer(
            input_sents,
            padding="longest",
            return_tensors="pt"
        )

        # Encode inputs
        input_ids = tokenized_sents.input_ids
        input_masks = tokenized_sents.attention_mask

        return input_ids, input_masks

    def _process_outputs(self, output_sents: List[str]):
        # update the prompts
        tokenized_sents = self.dec_tokenizer(
            output_sents,
            padding="longest",
            return_tensors="pt"
        )

        # Encode outputs
        output_ids = tokenized_sents.input_ids
        output_ids.masked_fill_(output_ids == self.dec_tokenizer.pad_token_id, -100)

        return output_ids

    def train(self):
        raise NotImplementedError("'train' function not implemented...")

    def train_step(self):
        raise NotImplementedError("'train_step' function not implemented...")

    def validate(self):
        raise NotImplementedError("'evaluate' function not implemented...")

    def validate_step(self):
        raise NotImplementedError(
            "'evaluate_step' function not implemented...")

    def test(self):
        raise NotImplementedError("'test' function not implemented...")

    def test_step(self):
        raise NotImplementedError("'test_step' function not implemented...")