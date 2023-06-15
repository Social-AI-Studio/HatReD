seed=$1
warmup=$2
lr=$3

python -u main.py \
    --SEED $seed \
    --ENC_MODEL_NAME "uclanlp/visualbert-vqa-coco-pre" \
    --DEC_MODEL_NAME "roberta-base" \
    --SAVE_FILEPATH "visualbert-roberta-$seed-$lr-$warmup" \
    --TEST_MODEL_FILEPATH "./fhm/visualbert-roberta-$seed-$lr-$warmup-best-harmonic-mean.pt" \
    --BATCH_SIZE 16 \
    --EPOCHS 30 \
    --GRADIENT_ACCUMULATION_STEPS 2 \
    --USE_ENTITIES \
    --USE_DEMOGRAPHICS \
    --CLEAN_CONTRACTIONS \
    --WARM_UP $warmup \
    --LR_RATE $lr \
    --TRAIN_ANNOTATIONS "../datasets/annotations/fhm_train.jsonl" \
    --TEST_ANNOTATIONS "../datasets/annotations/fhm_test.jsonl" \
    --FEATURES_FILEPATH "../datasets/features/clean/" \
    --TARGET_MODE 'random' \
    --CUDA_DEVICE cuda:0 
