# from .decoder import MemeDatasetGeneration, MemeDataset #, MemeDatasetRoBERTa2GPT2
from .captions import MemeCaptionDataset
from .features import MemeFeatureDataset

from typing import Optional

def get_datasets(
        train_annotations_fp: str, 
        test_annotations_fp: str,
        use_entities: bool, 
        use_demographics: bool, 
        use_understandings: bool,
        clean_contractions: bool,
        decoder_name: str,
        sep_token: str,
        captions_fp: Optional[str] = None,
        features_fp: Optional[str] = None,
    ):

    print('--- Filepaths ---')
    print('Captions filepath:', captions_fp)
    print('Features filepath:', features_fp)

    print('--- External Knowledge (Feature Extractors) ---')
    print('Adding entity information?', use_entities)
    print('Adding demographic information?', use_demographics)
    print('Adding understandings information?', use_understandings)

    print('--- NLP Cleaning Steps---')
    print('Clean Contractions?', clean_contractions)
    
    kwargs = {
        "use_entities": use_entities,
        "use_demographics": use_demographics,
        "use_understandings": use_understandings,
        "clean_contractions": clean_contractions,
        "decoder_name": decoder_name,
        "sep_token": sep_token
    }

    if captions_fp:
        train_dataset = MemeCaptionDataset(train_annotations_fp, captions_fp, **kwargs)
        test_dataset = MemeCaptionDataset(test_annotations_fp, captions_fp, **kwargs)
    elif features_fp:
        train_dataset = MemeFeatureDataset(train_annotations_fp, features_fp, **kwargs)
        test_dataset = MemeFeatureDataset(test_annotations_fp, features_fp, **kwargs)
    else:
        raise Exception("'captions_fp' and 'features_fp' cannot be None.")

    print('The length of the train dataset is:', len(train_dataset))
    print('The length of the train dataset is:', len(test_dataset))

    return train_dataset, test_dataset