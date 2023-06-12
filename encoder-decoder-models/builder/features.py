import os
import torch
import random
import contractions
import pickle as pkl
import numpy as np

from typing import List, Dict
from .base import MemeDataset

class MemeFeatureDataset(MemeDataset):

    def __init__(self,  
                 annotations_fp: str, 
                 features_dir: str,
                 use_entities: bool, 
                 use_demographics: bool,
                 clean_contractions: bool,
                 decoder_name: str,
                 sep_token: str):
        
        super().__init__(
            annotations_fp, use_entities, use_demographics,
            clean_contractions, sep_token
        )

        # Get target column
        self.decoder_name = decoder_name

        # Process Inputs
        self.visual_embeds = self.load_visual_embeds(features_dir)
        self.records = self.process_records()
    
    def load_visual_embeds(self, features_dir: str) -> dict:
        visual_embeds = {}
        for row in self.annotations:
            img_id = row['img'].split('.')[0]

            embeds_fp = os.path.join(features_dir, f"{img_id}.npy")
            embeds = np.load(embeds_fp)

            visual_embeds[img_id] = embeds

        return visual_embeds

    def process_records(self) -> List[object]:
        records = []
        for row in self.annotations:
            img_id = row['img'].split('.')[0]
            input_sent = row['text'].strip()
            output_sents = [r.lower().strip() for r in row['reasonings']]

            # clean contractions
            if self.clean_contractions:
                input_sent = contractions.fix(input_sent)

            # append external knowledge
            input_sent = self.append_external_feats(input_sent, row)
            input_sent = input_sent.replace(":", "")

            # prepare suffix (GPT2 requires manual insertion of suffix)
            suffix = " <|endoftext|>" if self.decoder_name == "gpt2" else ""
            output_sents = [
                x + suffix for x 
                in output_sents
            ]

            visual_embeds = self.visual_embeds[img_id]
            visual_embeds = visual_embeds[:36]


            # add to records
            records.append({
                'input_sent': input_sent,
                'visual_embeds': visual_embeds,
                'output_sents': output_sents,
            })

        return records
        
    def __getitem__(self,index):
        return self.records[index]