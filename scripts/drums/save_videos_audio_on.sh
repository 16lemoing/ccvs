#!/bin/bash

GPU_IDS=0
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/generator.py \
 --name save_videos_audio_on_drum \
 --x_which_iter "latest" --x_load_path "checkpoints/2021-05-20-21:33:55-transformer_audio_drums" \
 --load_vid_len 90 --max_vid_step 1 --vid_len 45 --vid_skip 16 --x_cond_len 960 --batch_size_vid 2 --batch_size_valid_mult 1 --num_workers 4 --n_iter 50 \
 --shuffle_valid --q_skip_context 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --q_skip_memory 15 --x_sample --x_top_k 100 --x_temperature 1.0 \
 --x_z_chunk 80 --keep_state --include_id --q_keep_first --q_n_first 8 \
 --dataset "drum" --max_dim 128 --is_seq --log_fps 1 \
 --gpu_ids ${GPU_IDS} \
 --q_z_num 1024 --q_z_size 512 --q_z_shape 8 8 \
 --q_lr 0.002 --q_beta1 0.0 --q_beta2 0.99 --q_gan_loss "logistic" \
 --q_use_enc --q_use_dec --q_use_di --q_use_vgg_img --q_use_gan_feat_img --q_use_direct_recovery_img \
 --q_necf 128 --q_necf_mult 1 1 2 2 4 --q_ndcf_mult 1 1 2 2 4 --q_ndcf 64 \
 --q_enc_model "skipgan" --q_dec_model "skipgan" --q_use_inter --q_inter_p 0.75 --q_use_ema \
 --a_stft_num 1024 --a_stft_size 512 --a_stft_hsize 512 --a_stft_shape 8 2 \
 --a_lr 0.002 --a_beta1 0.0 --a_beta2 0.99 \
 --x_z_num 1024 --x_z_len 1280 --x_n_layer 24 --x_n_head 16 --x_n_embd 1024 --x_lr 0.00001 \
 --x_num_blocks 16 --x_state_num 1024 --x_state_size 16 --x_stft \
 --x_emb_mode "temporal" \
 --q_which_iter "latest" --q_load_path "checkpoints/2021-05-18-23:29:37-frame_autoencoder_drums" \
 --a_which_iter "best" --a_load_path "checkpoints/2021-05-18-12:38:49-stft_autoencoder_drums"