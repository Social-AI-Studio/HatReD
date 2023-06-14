import torch
import random
import contractions
import pickle as pkl

from typing import List, Dict
from .base import MemeDataset

class MemeCaptionDataset(MemeDataset):

    def __init__(self,  
                 annotations_fp: str, 
                 captions_fp: str,
                 use_entities: bool, 
                 use_demographics: bool,
                 use_understandings: bool,
                 clean_contractions: bool,
                 decoder_name: str,
                 sep_token: str):
        
        super().__init__(
            annotations_fp, use_entities, use_demographics, use_understandings,
            clean_contractions, sep_token
        )

        # Get target column
        self.decoder_name = decoder_name

        # Process Inputs
        self.captions = self.load_captions(captions_fp)
        self.records = self.process_records()
    
    def load_captions(self, filepath: str):
        with open(filepath,'rb') as f:
            captions = pkl.load(f)

        return captions

    def process_records(self) -> List[object]:
        records = []
        for row in self.annotations:
            img_id = row['img'].split('.')[0]
            input_sent = row['text'].strip()
            output_sents = [r.lower().strip() for r in row['reasonings']]

            # clean contractions
            if self.clean_contractions:
                input_sent = contractions.fix(input_sent)

            # append captions
            cap = self.captions[img_id].strip()
            cap = cap[:-1] if cap[-1] == '.' else cap
            input_sent += f" {self.sep_token} {cap}"

            # append external knowledge
            input_sent = self.append_external_feats(input_sent, row)
            input_sent = input_sent.replace(":", "")

            # prepare suffix (GPT2 requires manual insertion of suffix)
            suffix = " <|endoftext|>" if self.decoder_name == "gpt2-large" else ""
            output_sents = [
                x + suffix for x 
                in output_sents
            ]


            # add to records
            records.append({
                'input_sent': input_sent,
                'output_sents': output_sents,
            })

        return records
        
    def __getitem__(self,index):
        return self.records[index]