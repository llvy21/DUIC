q=1
l=0.016
MODEL_PATH=./
CUDA_VISIBLE_DEVICES=0 python3 eval.py \
        --lambda ${l} \
        --quality ${q} \
        -m cheng2020-attn \
        --epochs 2000 \
        -lr 1e-3 \
        --cuda \
        --model_prefix $MODEL_PATH \
        --image ./test.png
