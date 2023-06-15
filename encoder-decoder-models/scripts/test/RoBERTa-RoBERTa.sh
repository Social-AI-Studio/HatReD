seed=$1
warmup=$2
lr=$3

python -u main.py \
    --SEED $seed \
    --ENC_MODEL_NAME "roberta-base" \
    --DEC_MODEL_NAME "roberta-base" \
    --SAVE_FILEPATH "roberta-roberta-$seed-$lr-$warmup-base" \
    --TEST_MODEL_FILEPATH "./fhm/roberta-roberta-base-$seed-$lr-$warmup-best-harmonic-mean.pt" \
    --BATCH_SIZE 16 \
    --EPOCHS 30 \
    --GRADIENT_ACCUMULATION_STEPS 2 \
    --USE_ENTITIES \
    --USE_DEMOGRAPHICS \
    --CLEAN_CONTRACTIONS \
    --WARM_UP $warmup \
    --LR_RATE $lr \
    --TRAIN_ANNOTATIONS "../datasets/hatred/annotations/fhm_train.jsonl" \
    --TEST_ANNOTATIONS "../datasets/hatred/annotations/fhm_test.jsonl" \
    --CAPTIONS_FILEPATH "../datasets/hatred/captions/fhm_clean_captions.pkl" \
    --TARGET_MODE 'random' \
    --CUDA_DEVICE cuda:0

