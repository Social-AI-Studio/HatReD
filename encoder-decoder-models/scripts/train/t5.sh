seed=$1
warmup=$2
lr=$3

python -u main.py \
    --SEED $seed \
    --ENC_MODEL_NAME "t5-large" \
    --DEC_MODEL_NAME "t5-large" \
    --SAVE_FILEPATH "t5-large-$seed-$lr-$warmup" \
    --USE_ENTITIES \
    --USE_DEMOGRAPHICS \
    --CLEAN_CONTRACTIONS \
    --WARM_UP $warmup \
    --EPOCHS 30 \
    --BATCH_SIZE 16 \
    --GRADIENT_ACCUMULATION_STEPS 2 \
    --TRAIN_ANNOTATIONS "../datasets/hatred/annotations/fhm_train.jsonl" \
    --TEST_ANNOTATIONS "../datasets/hatred/annotations/fhm_test.jsonl" \
    --CAPTIONS_FILEPATH "../datasets/hatred/captions/fhm_clean_captions.pkl" \
    --TARGET_MODE 'random' \
    --LR_RATE $lr \
    --CUDA_DEVICE cuda:0 \
    --TRAIN