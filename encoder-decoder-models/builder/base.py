import os
import json
import pickle as pkl

from typing import List

class MemeDataset:

    def __init__(self, 
                 annotations_fp: str,
                 use_entities: bool, 
                 use_demographics: bool, 
                 clean_contractions: bool,
                 sep_token: str):

        self.use_entities = use_entities
        self.use_demographics = use_demographics

        self.clean_contractions = clean_contractions
        self.sep_token = sep_token

        self.annotations = self.load_data(annotations_fp)

    def load_data(self, annotation_filepath):
        records_fp = os.path.join(annotation_filepath)
        
        records = []
        with open(records_fp, 'r') as f:
            for record in f.readlines():
                records.append(json.loads(record))

        return records

    def append_external_feats(
            self, 
            input_sent: str, 
            data: object
        ) -> str:
        for condition, column in (
            (self.use_entities, 'entity'),
            (self.use_demographics, 'race')
        ):
            if condition:
                input_sent += f" {self.sep_token} {data[column].strip()}"

        return input_sent

    def __getitem__(self, index):
        raise NotImplementedError("__getitem__ not implemented")
        
    def __len__(self):
        return len(self.records)
    
