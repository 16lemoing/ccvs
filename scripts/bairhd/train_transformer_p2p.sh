#!/bin/bash

GPU_IDS="0,1,2,3"
NUM_GPUS=4

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/transformer_trainer.py \
 --name transformer_p2p_bairhd \
 --dataset "bairhd" --max_dim 256 --is_seq --log_fps 1 --vid_len 16 \
 --save_latest_freq 1000 --num_workers 8 --gpu_ids ${GPU_IDS} \
 --n_iter 200000 --batch_size_vid 16 \
 --q_d_reg_every 16 \
 --q_z_num 1024 --q_z_size 512 --q_z_shape 8 8 \
 --q_lr 0.002 --q_beta1 0.0 --q_beta2 0.99 --q_gan_loss "logistic" \
 --q_use_enc --q_use_dec --q_use_di --q_use_vgg_img --q_use_gan_feat_img --q_use_direct_recovery_img \
 --q_necf 128 --q_necf_mult 1 1 2 2 4 4 --q_ndcf_mult 1 1 2 2 4 4 --q_ndcf 64 \
 --q_enc_model "skipgan" --q_dec_model "skipgan" --q_use_inter --q_inter_p 0.75 --q_use_ema \
 --x_z_num 1024 --x_z_len 1024 --x_cond_len 64 --x_n_layer 24 --x_n_head 16 --x_n_embd 1024 --x_lr 0.00001 --x_sample --x_top_k 100 \
 --x_z_chunk 64 --x_p2p --p2p_len 16 --x_emb_mode "temporal" \
 --q_which_iter "latest" --q_load_path "checkpoints/YOUR_AUTOENCODER_CHECKPOINT_FOLDER_HERE"