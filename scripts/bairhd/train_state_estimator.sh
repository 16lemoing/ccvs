#!/bin/bash

GPU_IDS=0
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/state_estimator_trainer.py \
 --name state_estimator_bairhd \
 --dataset "bairhd" --max_dim 256 \
 --save_latest_freq 100 --num_workers 8 --gpu_ids ${GPU_IDS} --log_freq 100 --n_iter_eval 100 \
 --n_iter 10000 --batch_size_img 64 \
 --q_z_num 1024 --q_z_size 512 --q_z_shape 8 8 \
 --q_lr 0.002 --q_beta1 0.0 --q_beta2 0.99 --q_gan_loss "logistic" \
 --q_use_enc --q_use_dec --q_use_di --q_use_vgg_img --q_use_direct_recovery_img \
 --q_necf 128 --q_necf_mult 1 1 2 2 4 4 --q_ndcf_mult 1 1 2 2 4 4 --q_ndcf 64 \
 --q_enc_model "skipgan" --q_dec_model "skipgan" --q_use_inter \
 --q_slide_inter --q_inter_p 0.75 \
 --q_use_elastic_flow_recovery --q_use_ema \
 --q_use_dv --q_use_vgg_vid --q_use_direct_recovery_vid --q_skip_memory 4 --q_skip_context 1 2 3 4 \
 --q_which_iter "latest" --q_load_path "checkpoints/YOUR_AUTOENCODER_CHECKPOINT_FOLDER_HERE" \
 --load_state --s_state_size 2 --s_state_num 128 --s_lr 0.01 \
 --shuffle_valid
