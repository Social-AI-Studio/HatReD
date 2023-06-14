import os

from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast

# project_dir = Path(__file__).resolve().parent.parent  # VLT5
# workspace_dir = project_dir.parent
# dataset_dir = workspace_dir.joinpath('datasets/').resolve()
# coco_dir = dataset_dir.joinpath('COCO')
# vg_dir = dataset_dir.joinpath('VG')

import contractions

dataset_dir = "../../datasets/"
features_fp = os.path.join(dataset_dir, "features/clean/{img_id}.npy")
features_info_fp = os.path.join(dataset_dir, "features/clean/{img_id}_info.npy")
annotations_fp = os.path.join(dataset_dir, "annotations/fhm_{split}_reasonings.jsonl")

class FHMFineTuneDataset(Dataset):
    def __init__(self, split='train', rank=-1, topk=-1, verbose=True, args=None):
        super().__init__()

        self.topk = topk
        self.verbose = verbose
        self.args = args

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data source: ', self.source)


        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if 't5' in self.args.tokenizer:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
        elif 'bart' in self.args.tokenizer:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
            special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        if self.args.oscar_tags:
            # Load VG Classes
            vg_classes = []
            with open(vg_dir.joinpath('objects_vocab.txt')) as f:
                for obj in f.readlines():
                    vg_classes.append(obj.split(',')[0].lower().strip())
            self.vg_classes = vg_classes

        # load annotations
        filepath = annotations_fp.format(split=split)
        self.data = []
        with open(filepath, 'r') as f:
            for record in f.readlines():
                self.data.append(json.loads(record))

        if self.verbose:
            print(f"{self.source} has {len(self.data)} images")
            print(f"Loaded {len(self.data)} data from", split)

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank
        if self.topk > 0:
            self.data = self.data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        if self.verbose:
            print("# all sentences:", len(self.data))

        # load features
        self.features = {}
        self.features_info = {}
        for record in self.data:
            # format filepath
            img = record['img']
            img_id = img.split('.')[0]
            feats_filepath = features_fp.format(img_id=img_id)
            info_filepath = features_info_fp.format(img_id=img_id)

            # load features
            feats = np.load(feats_filepath)
            self.features[img] = feats

            # load features info
            info = np.load(info_filepath, allow_pickle=True).item()
            self.features_info[img] = info
        
        print(f"Loaded {len(self.features)} feature data")

    def __len__(self):
        return len(self.data)

    def _clip_dim(self, col: np.ndarray, max_val: int):
        col[col > max_val] = max_val

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        ###### Image ######
        if self.args.use_vision:
            img = datum['img']
            out_dict['img'] = img

            # Normalize the boxes (to 0 ~ 1)
            img_h = self.features_info[img]['image_height']
            img_w = self.features_info[img]['image_width']
            boxes = self.features_info[img]['bbox']

            # # clip coordinates to width and height
            # self._clip_dim(boxes[:, 0], img_w)
            # self._clip_dim(boxes[:, 2], img_w)
            # self._clip_dim(boxes[:, 1], img_h)
            # self._clip_dim(boxes[:, 3], img_h)

            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            # np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(-boxes, 0+1e-5)
            
            boxes = torch.from_numpy(boxes)
            boxes.clamp_(min=0.0, max=1.0)

            n_boxes = len(boxes)

            # load features
            feats = self.features[img]
            feats = torch.from_numpy(feats)

            n_boxes = min(n_boxes, self.args.max_n_boxes)
            out_dict['n_boxes'] = n_boxes
            # if not self.args.BUTD100:
            boxes = boxes[:n_boxes]
            feats = feats[:n_boxes]
            out_dict['boxes'] = boxes
            out_dict['vis_feats'] = feats

        ###### Text #####
        input_text = datum['text']
        input_text = contractions.fix(input_text)

        for column in ['entity', 'race']:
            input_text += f" </s> {datum[column].strip()}" # TODO: BART incompatible

        if 't5' in self.args.tokenizer:
            input_ids = self.tokenizer.encode(
                input_text,
                max_length=self.args.max_text_length, truncation=True)
        elif 'bart' in self.args.tokenizer:
            input_ids = self.tokenizer.encode(
                input_text,
                max_length=self.args.max_text_length, truncation=True)
        else:
            input_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(input_text)[:self.args.max_text_length - 1] + ['[SEP]'])

        out_dict['input_text'] = input_text
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)

        # Reference sentences
        out_dict['targets'] = [r.lower().strip() for r in datum['reasonings']]
        
        if self.source == "train":
            # reasoning
            output_text = random.sample(out_dict['targets'], 1)[0]

            if 't5' in self.args.tokenizer:
                target_ids = self.tokenizer.encode(output_text, max_length=self.args.gen_max_length, truncation=True)
            elif 'bart' in self.args.tokenizer:
                target_ids = self.tokenizer.encode(output_text, max_length=self.args.gen_max_length, truncation=True)

            assert len(target_ids) <= self.args.gen_max_length, len(target_ids)
            out_dict['output_text'] = output_text
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

        # if 'targets' in datum:
        #     out_dict['targets'] = datum['targets']

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        if self.args.use_vision:
            V_L = max(entry['n_boxes'] for entry in batch)
            # V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
            vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        # sentences = []

        targets = []
        imgs = []
        img_paths = []
        input_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            if self.args.use_vision:
                n_boxes = entry['n_boxes']
                boxes[i, :n_boxes] = entry['boxes']
                vis_feats[i, :n_boxes] = entry['vis_feats']
                vis_attention_mask[i, :n_boxes] = 1
                imgs.append(entry['img'])
                # img_paths.append(entry['img_path'])

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'input_text' in entry:
                input_text.append(entry['input_text'])

            # sentences.append(entry['sent'])

            if 'targets' in entry:
                targets.append(entry['targets'])


        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids

        if self.args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            batch_entry['vis_attention_mask'] = vis_attention_mask
            batch_entry['img'] = imgs
            batch_entry['img_paths'] = img_paths

        # batch_entry['sent'] = sentences

        batch_entry['input_text'] = input_text

        batch_entry['targets'] = targets

        batch_entry['task'] = 'meme_reasonings'

        return batch_entry


def get_loader(args, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    verbose = (gpu == 0)

    dataset = FHMFineTuneDataset(
        split,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args)

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=workers, pin_memory=True, 
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=False)


    if verbose:
        if mode == "train":
            loader.evaluator = FHMTrainEvaluator()
        else:
            loader.evaluator = FHMTestEvaluator()

    loader.task = 'meme_reasonings'

    return loader

#     if verbose:
#         loader.evaluator = FHMEvaluator()

#     loader.task = 'meme_reasonings'

#     return loader

# from evaluate import load

# class FHMEvaluator:
#     def __init__(self):
#         self.bleu = load('bleu')
#         self.rouge = load('rouge')
#         self.bertscore = load('bertscore')
#         self.meteor = load('meteor')

#     def evaluate(self, predicts, answers):

#         results = self.evaluator.run_evaluation(predicts, answers)

#         return results

#         import os

from evaluate import load

class FHMTrainEvaluator:
    def __init__(self):
        pass

    def evaluate(self, predicts, answers):
        return {}

class FHMTestEvaluator:
    def __init__(self):
        self.bleu = load('bleu')
        self.rouge = load('rouge')
        self.bertscore = load('bertscore')
        self.meteor = load('meteor')

    def evaluate(self, predicts, answers):

        # compute metrics
        bertscore_results = self.bertscore.compute(predictions=predicts, references=answers, rescale_with_baseline=True, lang="en")
        bleu_results = self.bleu.compute(predictions=predicts, references=answers)
        rouge_results = self.rouge.compute(predictions=predicts, references=answers)
        meteor_results = self.meteor.compute(predictions=predicts, references=answers)

        results = {
            "candidates": predicts,
            "references": answers
        }

        # bleu
        results['bleu'] = bleu_results['bleu']
        results['bleu-1'] = bleu_results['precisions'][0]
        results['bleu-2'] = bleu_results['precisions'][1]
        results['bleu-3'] = bleu_results['precisions'][2]
        results['bleu-4'] = bleu_results['precisions'][3]
        results['brevity-penalty'] = bleu_results['brevity_penalty']

        # rouge
        results = {**results, **rouge_results}
        
        # BERTScore
        avg = lambda lst: sum(lst) / len(lst)
        results['P-BERT'] = avg(bertscore_results['precision'])
        results['R-BERT'] = avg(bertscore_results['recall'])
        results['F-BERT'] = avg(bertscore_results['f1'])

        # meteor
        results['meteor'] = meteor_results['meteor']

        # harmonic mean
        harmonic_mean = 2 * (results['bleu'] * results['rougeL']) / (results['bleu'] + results['rougeL'])
        results['harmonic-mean'] = harmonic_mean
        

        return results

