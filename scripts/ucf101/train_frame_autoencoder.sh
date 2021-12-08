#!/bin/bash

GPU_IDS="0,1,2,3"
NUM_GPUS=4

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/frame_autoencoder_trainer.py \
 --name frame_autoencoder_ucf101 \
 --dataset "ucf101" --max_dim 256 --is_seq \
 --save_latest_freq 1000 --num_workers 16 --gpu_ids ${GPU_IDS} --log_freq 2000 \
 --n_iter 800000 --batch_size_img 96 --batch_size_vid 16 \
 --q_d_reg_every 16 \
 --q_z_num 1024 --q_z_size 512 --q_z_shape 8 8 \
 --q_lr 0.002 --q_beta1 0.0 --q_beta2 0.99 --q_gan_loss "logistic" \
 --q_use_enc --q_use_dec --q_use_di --q_use_vgg_img --q_use_gan_feat_img --q_use_direct_recovery_img \
 --q_necf 128 --q_necf_mult 1 1 2 2 4 4 --q_ndcf_mult 1 1 2 2 4 4 --q_ndcf 64 \
 --q_enc_model "skipgan" --q_dec_model "skipgan" --q_use_inter \
 --n_consecutive_img 2 --img_out_of_n 30 --q_slide_inter --q_inter_p 0.75 \
 --load_elastic_view --q_use_elastic_flow_recovery --elastic_alpha 3 --elastic_sigma 0.1 --elastic_min_zoom 0.9 --elastic_max_zoom 1.1 \
 --elastic_corruption --q_use_ema --blur_first 0.0 2.0 --distort_first \
 --q_use_dv --q_use_vgg_vid --q_use_direct_recovery_vid --load_vid_len 30 --vid_len 4 --q_skip_memory 4 --q_skip_context 1 2 3 4