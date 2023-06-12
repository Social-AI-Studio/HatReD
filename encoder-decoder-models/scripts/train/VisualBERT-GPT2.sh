seed=$1
warmup=$2
lr=$3

python -u main.py \
    --SEED $seed \
    --ENC_MODEL_NAME "uclanlp/visualbert-vqa-coco-pre" \
    --DEC_MODEL_NAME "gpt2" \
    --SAVE_FILEPATH "visualbert-gpt2-base-$seed-$lr-$warmup" \
    --USE_ENTITIES \
    --USE_DEMOGRAPHICS \
    --CLEAN_CONTRACTIONS \
    --WARM_UP $warmup \
    --EPOCHS 30 \
    --BATCH_SIZE 16 \
    --GRADIENT_ACCUMULATION_STEPS 2 \
    --TRAIN_ANNOTATIONS "../datasets/annotations/fhm_train_reasonings.jsonl" \
    --TEST_ANNOTATIONS "../datasets/annotations/fhm_test_reasonings.jsonl" \
    --FEATURES_FILEPATH "../datasets/features/clean/" \
    --TARGET_MODE 'random' \
    --LR_RATE $lr \
    --CUDA_DEVICE cuda:0 \
    --TRAIN