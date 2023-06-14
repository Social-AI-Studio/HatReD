import os
import math
import json
import torch
import random

from tqdm import tqdm
from .base import BaseTrainer
from .utils import get_epoch_summary, get_overall_summary


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class EncoderDecoderTrainer(BaseTrainer):

    def __init__(self, model, model_filename, enc_tokenizer, dec_tokenizer, 
                 train_loader, valid_loader, test_loader,
                 optim, scheduler, sep_token, target_mode,
                 gradient_accumulation_steps, device, epoches, logger):
        
        super().__init__(model, model_filename, enc_tokenizer, dec_tokenizer, 
                 train_loader, valid_loader, test_loader,
                 optim, scheduler, sep_token, target_mode,
                 gradient_accumulation_steps, device, epoches, logger)

    def train(self):
        for ep in range(self.epoches):
            self.logger.write(f"Epoch {ep}")
            self.train_iter(ep)
            self.validate_iter()
        
        # report overall results
        summary = get_overall_summary(self.checkpoint)
        self.logger.write(summary)

    def train_iter(self, epoch: int):
        self.model.train()

        total_loss = 0.0
        total_step = epoch * math.ceil(len(self.train_loader) / self.gradient_accumulation_steps)
        with tqdm(self.train_loader) as progressbar:
            for step, batch in enumerate(progressbar):
                current_step = step + 1

                step_optim = (
                    ((current_step % self.gradient_accumulation_steps) == 0) or
                    (current_step == len(self.train_loader))
                )

                loss = self.train_step(batch, step_optim)

                if step_optim:
                    total_step += 1
                
                # update loss postfix
                current_lr = self.scheduler.get_last_lr()[0]
                progressbar.set_postfix(step=total_step, lr=current_lr, loss=loss)

                total_loss += loss

            self.logger.write(f"Average Training Loss: {total_loss}")

    def train_step(self, batch: dict, step_optim: bool):
        input_sents = batch['input_sent']
        output_sents = batch['output_sents']

        # Select the target sentence
        if self.target_mode == "first":
            target_sent = [x[0] for x in output_sents]
        elif self.target_mode == "random":
            target_sent = [
                random.sample(x, 1)[0] for x in output_sents
            ]
        else:
            raise Exception("Unknown target_mode")

        # insert separator tokens
        input_sents = [x.format(sep_token=self.sep_token) for x in input_sents]

        # prepare inputs
        input_ids, attention_mask = self._process_inputs(input_sents)
        output_ids = self._process_outputs(target_sent)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        output_ids = output_ids.to(self.device)
        
        # run through the model
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=output_ids
        )

        # compute loss
        loss = model_outputs.loss / self.gradient_accumulation_steps
        loss.backward()

        if step_optim:
            self.optim.step()
            self.scheduler.step()
            self.optim.zero_grad()
        
        return loss.item()

    def validate_iter(self):
        self.model.eval()

        total_candidates, total_references = [], []
        for _, batch in enumerate(tqdm(self.valid_loader)):
            pred_tokens = self.validate_step(batch)
            pred_tokens = pred_tokens.detach().cpu()

            # decode the generation
            candidates = self.dec_tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)

            total_candidates += candidates
            total_references += [
                [x.replace('<|endoftext|>', '') for x in lst] 
                for lst in batch['output_sents']
            ]

        print(f"candidate: {total_candidates[0]}")
        print(f"refrence: {total_references[0]}")
        
        # compute metric 
        results = self.evaluator.compute_metrics(total_candidates, total_references)
        
        # report epoch results
        msg = get_epoch_summary(results)
        self.logger.write(msg)
        
        # save model
        self.checkpoint.update_checkpoint(results)


    def validate_step(self, batch: dict):
        with torch.no_grad():
            input_sents = batch['input_sent']
            output_sents = batch['output_sents']
            output_sents = [[x.replace('<|endoftext|>', '') for x in lst] for lst in output_sents]

            # prepare inputs
            input_ids, attention_masks = self._process_inputs(input_sents )
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            
            # run through the model
            greedy_strategy = {
                "do_sample": False,
                "max_length": 50,
                "num_beams": 1
            }

            return self.model.generate(
                input_ids,
                attention_mask=attention_masks,
                **greedy_strategy
            )

    def test(self):
        for name, kwargs in self.decoding_strategies:
            results = self.test_iter(kwargs)
            summary = get_epoch_summary(results, f"Overall Performance ({name})")
            
            # report results
            self.logger.write(summary)

            # write all candidates into a single file
            filename = os.path.join(f"{self.model_filename}-{name}-candidates.txt")
            with open(filename, 'w') as f:
                json.dump(results['candidates'], f)

            # write all candidates into a single file
            filename = os.path.join(f"{self.model_filename}-references.txt")
            with open(filename, 'w') as f:
                json.dump(results['references'], f)

    
    def test_iter(self, decoding_strategy):
        self.model.eval()

        total_candidates, total_references = [], []
        for _, batch in enumerate(tqdm(self.valid_loader)):
            pred_tokens = self.test_step(batch, decoding_strategy)

            # decode the generation
            candidates = self.dec_tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)

            total_candidates += candidates
            total_references += [
                [x.replace('<|endoftext|>', '') for x in lst] 
                for lst in batch['output_sents']
            ]
        
        results = self.evaluator.compute_metrics(total_candidates, total_references)
        results['candidates'] = total_candidates
        results['references'] = total_references

        return results

    def test_step(self, batch: dict, decoding_strategy: dict):
        with torch.no_grad():
            input_sents = batch['input_sent']
            output_sents = batch['output_sents']
            output_sents = [[x.replace('<|endoftext|>', '') for x in lst] for lst in output_sents]

            # prepare inputs
            input_ids, attention_masks = self._process_inputs(input_sents )
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            
            # set_seed(9620)

            return self.model.generate(
                input_ids,
                attention_mask=attention_masks,
                **decoding_strategy
            )