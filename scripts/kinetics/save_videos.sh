#!/bin/bash

GPU_IDS=0
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/generator.py \
 --name save_videos_kinetics600 \
 --x_which_iter "latest" --x_load_path "checkpoints/2021-05-10-22:25:50-transformer_kinetics600" \
 --vid_len 16 --vid_skip 30 --x_cond_len 320 --batch_size_vid 16 --batch_size_valid_mult 1 --num_workers 8 --n_iter 78 \
 --x_sample --x_top_k 100 --x_temperature 1.0 --shuffle_valid --q_skip_context 1 2 3 4 5 6 7 8 --q_skip_memory 8 \
 --dataset "kinetics600" --max_dim 64 --is_seq --log_fps 1 \
 --data_specs 64p_square_32t --load_data --num_folds_train 100 --random_fold_train \
 --gpu_ids ${GPU_IDS} \
 --q_z_num 16384 --q_z_size 512 --q_z_shape 8 8 \
 --q_lr 0.002 --q_beta1 0.0 --q_beta2 0.99 --q_gan_loss "logistic" \
 --q_use_enc --q_use_dec --q_use_di --q_use_vgg_img --q_use_gan_feat_img --q_use_direct_recovery_img \
 --q_necf 256 --q_necf_mult 1 1 2 2 --q_ndcf_mult 1 1 2 2 \
 --q_enc_model "skipgan" --q_dec_model "skipgan" --q_use_inter --q_inter_p 0.75 --q_use_ema \
 --x_z_num 16384 --x_z_len 1024 --x_n_layer 24 --x_n_head 16 --x_n_embd 1024 --x_lr 0.00001 \
 --x_z_chunk 64 --x_emb_mode "temporal" \
 --q_which_iter "latest" --q_load_path "checkpoints/2021-05-04-13:59:21-frame_autencoder_kinetics600" \
