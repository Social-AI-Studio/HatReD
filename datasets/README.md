# HatReD Dataset

As our research expands upon Facebook AI's Hateful Memes dataset through supplementary annotations, specifically reasoning annotations, we kindly request that interested researchers duly acknowledge and adhere to Facebook AI's Hateful Memes dataset licence agreements. This entails the requisite download of the original dataset provided by Facebook AI.

## Step 1. Review and Accept Facebook AI's Dataset Licence Agreement
Researchers may access the Hateful Memes dataset license agreements by visiting the official website at [https://hatefulmemeschallenge.com/](https://hatefulmemeschallenge.com/). Once researchers have carefully reviewed and duly accepted the terms outlined in the license agreements, they are eligible to proceed with the download of the Hateful Memes datasets. This includes
- train, dev, dev_seen and test annotations
- images **(critical for vision-language multimodal models)**

## Step 2. Clone the Facebook AI's Hateful Memes Fine-Grained Challenge Repository
Facebook AI has recently broadened the scope of their annotations to encompass a wider range of classification tasks, specifically addressing protected attack and protected target classification. Through our own investigations, we have observed that this expansion has rectified inaccuracies in the annotations previously mislabeled as hateful/non-hateful.

Researchers conveniently obtain these updated annotations by cloning the repository using the following link: https://github.com/facebookresearch/fine_grained_hateful_memes.git.

## Step 3. Constructing the HatReD's train and test set
To construct the train and test dataset for HatReD using the downloaded Hateful Memes Finegrained annotations, you can execute the following command.

```bash
python3 datasets/create_hatred_dataset.py --github-dir /path/to/repository
```

# MAMI Dataset (For Human Evaluation)
Coming soon...