seed=$1
warmup=$2
lr=$3

python -u main.py \
    --SEED $seed \
    --ENC_MODEL_NAME "t5-large" \
    --DEC_MODEL_NAME "t5-large" \
    --SAVE_FILEPATH "t5-large-$seed-$lr-$warmup" \
    --TEST_MODEL_FILEPATH "./fhm/t5-large-$seed-$lr-$warmup-best-harmonic-mean.pt" \
    --BATCH_SIZE 16 \
    --EPOCHS 30 \
    --GRADIENT_ACCUMULATION_STEPS 2 \
    --USE_ENTITIES \
    --USE_DEMOGRAPHICS \
    --CLEAN_CONTRACTIONS \
    --WARM_UP $warmup \
    --LR_RATE $lr \
    --TRAIN_ANNOTATIONS "../datasets/annotations/fhm_train_reasonings.jsonl" \
    --TEST_ANNOTATIONS "../datasets/annotations/fhm_test_reasonings.jsonl" \
    --CAPTIONS_FILEPATH "../datasets/captions/fhm_clean_captions.pkl" \
    --TARGET_MODE 'random' \
    --CUDA_DEVICE cuda:0 