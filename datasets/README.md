# HatReD Dataset
As our research expands upon Facebook AI's Hateful Memes dataset through supplementary annotations, specifically reasoning annotations, we kindly request that interested researchers duly acknowledge and adhere to Facebook AI's Hateful Memes dataset licence agreements. This entails the requisite download of the original dataset provided by Facebook AI.

## Step 1. Review and Accept Facebook AI's Dataset Licence Agreement
Researchers may access the Hateful Memes dataset license agreements by visiting the official website at [https://hatefulmemeschallenge.com/](https://hatefulmemeschallenge.com/). Once researchers have carefully reviewed and duly accepted the terms outlined in the license agreements, they are eligible to proceed with the download of the Hateful Memes datasets. This includes
- train, dev, dev_seen and test annotations
- images **(critical for vision-language multimodal models)**

## Step 2. Clone the Facebook AI's Hateful Memes Fine-Grained Challenge Repository
Facebook AI has recently broadened the scope of their annotations to encompass a wider range of classification tasks, specifically addressing protected attack and protected target classification. Through our own investigations, we have observed that this expansion has rectified inaccuracies in the annotations previously mislabeled as hateful/non-hateful.

Researchers conveniently obtain these updated annotations by cloning the repository using the following link: https://github.com/facebookresearch/fine_grained_hateful_memes.git.

## Step 3. Cleaning Meme Images (Optional)
The image's quality has the potential to impact the subsequent preprocessing stages, including entity extraction, demographic identification, captioning, and embedding extraction. To address this concern, an additional measure was taken to eliminate text overlays from memes. Precisely, we utilized the image cleaning procedure outlined in HimariO's [repository](https://github.com/HimariO/HatefulMemesChallenge/blob/main/data_utils/README.md), employing DeepFillV2 from MMEdit for image in-painting.

Due to the dataset licencing agreement, we are unable to provide direct downloads to the clean meme images.

## Step 4. Image Captioning (Provided)
In order to engage with text-based multimodal models, we employed an image captioning technique to derive the textual representation that portrays the image. To accomplish this objective, we utilized the [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) model for the extraction of image captions.

The extracted image captions can be found in the following directory: `fhm/captions`

## Step 5. Extracting Web Entities and Demographic Information (Provided)
To enhance the visual attributes available for text-based multimodal models, we employed the FairFace classifier and Google Web Vision APIs, specifically the Web Detect API, for the purpose of race and entity extraction, respectively.

The extracted image captions can be found in the following directory: `fhm/auxiliary`

## Step 6. Extracting Visual Embeddings (Provided)
As VisualBERT and VL-T5 uses pre-extracted visual embeddings for training of their models, we have uploaded the extracted visual embeddings used for these models

The extracted visual embeddings can be found [here](). Note that the visual embeddings should be placed under `fhm/features/clean` folder. 

## Step 7. Constructing the HatReD's train and test set
To construct the train and test dataset for HatReD using the downloaded Hateful Memes Finegrained annotations, you can execute the following command.

```bash
python3 datasets/create_hatred_dataset.py --github-dir /path/to/repository
```