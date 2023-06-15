# Feature extraction

We use [Hao Tan's Detectron2 implementation of 'Bottom-up feature extractor'](https://github.com/airsplay/py-bottom-up-attention), which is compatible with [the original Caffe implementation](https://github.com/peteanderson80/bottom-up-attention).

Following LXMERT, we use the feature extractor which outputs 36 boxes per image.
We store features in hdf5 format.


## Download features

Download `datasets` folder from [Google Drive](https://drive.google.com/drive/folders/1MBBhlkP83VMKS2Qe0SmFfzkHhMpIG5wf?usp=sharing)


## Install feature extractor (optional)

Please follow [the original installation guide](https://github.com/airsplay/py-bottom-up-attention#installation).

## Manually extract & convert features (optional)

* `_prpoposal.py`: extract features from 36 detected boxes
* `_gt.py`: extract features from ground truth boxes
* `_mattnet.py`: extract features from box predictions shared from [MattNet](https://github.com/lichengunc/MAttNet#pre-computed-detectionsmasks)

```bash
# Pretrain/VQA: Download LXMERT's COCO features (tsv) and convert to hdf5
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip
python tsv_to_h5.py --tsv_path train2014_obj36.tsv --h5_path train2014_obj36.h5
python tsv_to_h5.py --tsv_path val2014_obj36.tsv --h5_path val2014_obj36.h5
# Get resplit_val_obj36.h5 from val2014_obj36.h5
python coco_val_compact.py

# Pretrain(VG)/GQA: Download LXMERT's VG features (tsv) and convert to hdf5
wget https://nlp.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/vg_gqa_obj36.zip
python tsv_to_h5.py --tsv_path vg_gqa_obj36.tsv --h5_path vg_gqa_obj36.h5

# RefCOCOg
python refcocog_gt.py --split train
python refcocog_mattnet.py --split val
python refcocog_mattnet.py --split test

# NLVR2: Download LXMERT's COCO features (tsv) and convert to hdf5
wget https://nlp.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/train_obj36.zip
wget https://nlp.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/valid_obj36.zip
wget https://nlp.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/test_obj36.zip
python tsv_to_h5.py --tsv_path train_obj36.tsv --h5_path train_obj36.h5
python tsv_to_h5.py --tsv_path valid_obj36.tsv --h5_path valid_obj36.h5
python tsv_to_h5.py --tsv_path test_obj36.tsv --h5_path test_obj36.h5

# Multi30K
# Download images following https://github.com/multi30k/dataset
python flickr30k_proposal.py --split trainval
python flickr30k_proposal.py --split test2017
python flickr30k_proposal.py --split test2018
```