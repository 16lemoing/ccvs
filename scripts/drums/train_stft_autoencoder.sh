#!/bin/bash

GPU_IDS=0
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/stft_autoencoder_trainer.py \
 --name stft_autoencoder_drums \
 --dataset "drums" --max_dim 128 \
 --save_latest_freq 500 --num_workers 8 --gpu_ids ${GPU_IDS} --log_freq 500 --n_iter_eval 500 \
 --n_iter 100000 --batch_size_vid 64 --vid_len 16 --load_vid_len 90 --vid_skip 10 \
 --a_stft_num 1024 --a_stft_size 512 --a_stft_hsize 512 --a_stft_shape 8 2 \
 --a_lr 0.002 --a_beta1 0.0 --a_beta2 0.99
