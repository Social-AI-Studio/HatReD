# The name of experiment
name=VLT5
seed=1111
lr=5e-5
warmup=0.2

output="snap/FHM/${name}_${seed}_${lr}_${warmup}"

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/fhm.py \
        --distributed --multiGPU \
        --train train \
        --valid test \
        --test test \
        --optim adamw \
        --warmup_ratio ${warmup} \
        --clip_grad_norm 5 \
        --lr $lr \
        --epochs 30 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output \
        --load snap/pretrain/VLT5/Epoch30 \
        --num_beams 1 \
        --batch_size 16 \
        --max_text_length 128 \
        --warmup_ratio ${warmup} \
        --max_n_boxes 36 \
        --seed $seed