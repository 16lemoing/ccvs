import os
import pickle
import datetime
import argparse
from argparse import Namespace

from tools import utils
from data.cat import KINETICS600_CAT

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser = self.initialize_base(parser)
        parser = self.initialize_base(parser, prefix="xb_")
        parser = self.initialize_qvid_generator(parser)
        parser = self.initialize_transformer(parser)
        parser = self.initialize_state_estimator(parser)
        parser = self.initialize_stft_ae(parser)
        self.initialized = True
        return parser

    def initialize_base(self, parser, prefix=""):
        # experiment specifics
        parser.add_argument(f'--{prefix}name', type=str, required=(prefix == ""), help='name of the experiment, it indicates where to store samples and models')
        parser.add_argument(f'--{prefix}gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument(f'--{prefix}phase', type=str, default='train', help='train, val, test, etc')

        # for input / output sizes
        parser.add_argument(f'--{prefix}batch_size_img', type=int, default=1, help='input batch size for images')
        parser.add_argument(f'--{prefix}n_consecutive_img', type=int, default=1, help='number of consecutive images within batch which belong to the same video')
        parser.add_argument(f'--{prefix}img_out_of_n', type=int, default=1, help='n_consecutive_img are taken from img_out_of_n video frames')
        parser.add_argument(f'--{prefix}load_elastic_view', type=str2bool, nargs='?', const=True, default=False, help='if specified, load elastic deformation view for image and corresponding flow')
        parser.add_argument(f'--{prefix}elastic_alpha', type=float, default=1.5, help='alpha for elastic view')
        parser.add_argument(f'--{prefix}elastic_sigma', type=float, default=0.15, help='sigma for elastic view')
        parser.add_argument(f'--{prefix}elastic_min_zoom', type=float, default=1.0, help='zoom (prop of original image)')
        parser.add_argument(f'--{prefix}elastic_max_zoom', type=float, default=1.0, help='zoom (prop of original image)')
        parser.add_argument(f'--{prefix}elastic_occlusion', type=str2bool, nargs='?', const=True, default=False, help='combine two elastic transformations')
        parser.add_argument(f'--{prefix}elastic_corruption', type=str2bool, nargs='?', const=True, default=False, help='generate corruption mask')
        parser.add_argument(f'--{prefix}elastic_mean_corruption', type=float, default=0.5, help='proportion of the image to be corrupted')
        parser.add_argument(f'--{prefix}distort_first', type=str2bool, nargs='?', const=True, default=False, help='distort context image instead of other')
        parser.add_argument(f'--{prefix}blur_first', type=float, nargs="+", default=None, help='blur image by specifying min and max sigma')
        parser.add_argument(f'--{prefix}batch_size_vid', type=int, default=1, help='input batch size for videos')
        parser.add_argument(f'--{prefix}batch_size_valid_mult', type=int, default=1, help='multiplication factor for validation batch size')
        parser.add_argument(f'--{prefix}true_dim', type=int, default=1024, help='resolution of saved images, or after being resized if that is the case')
        parser.add_argument(f'--{prefix}hr_dim', type=int, default=None, help='resolution to load raw images')
        parser.add_argument(f'--{prefix}max_dim', type=int, default=512, help='resolution up to which we wish to train our models')
        parser.add_argument(f'--{prefix}guide_dim', type=int, default=None, help='resolution of guides if different from max dim')
        parser.add_argument(f'--{prefix}dim', type=int, default=-1, help='resolution at which to initialize training (has no effect for the seg generator)')
        parser.add_argument(f'--{prefix}true_ratio', type=float, default=1.0, help='ratio width/height of saved images, final width will be max_dim * aspect_ratio')
        parser.add_argument(f'--{prefix}aspect_ratio', type=float, default=2.0, help='target width/height ratio')
        parser.add_argument(f'--{prefix}transpose', type=str2bool, nargs='?', const=True, default=False, help='transpose the input seg/img')
        parser.add_argument(f'--{prefix}imagenet_norm', type=str2bool, nargs='?', const=True, default=False, help='normalize images the same way as it is done for imagenet')
        parser.add_argument(f'--{prefix}colorjitter', type=float, default=None, help='randomly change the brightness, contrast and saturation of images')
        parser.add_argument(f'--{prefix}is_double_pendulum', type=str2bool, nargs='?', const=True, default=False, help='for pendulum dataset')

        # for setting inputs
        parser.add_argument(f'--{prefix}dataroot', type=str, default='./datasets/youtube_faces/')
        parser.add_argument(f'--{prefix}dataset', type=str, default='youtube_faces')
        parser.add_argument(f'--{prefix}load_extra', type=str2bool, nargs='?', const=True, default=False, help='useful when datasets contain extra train set')
        parser.add_argument(f'--{prefix}num_folds_train', type=int, default=None, help='if specified, only load data fold by fold')
        parser.add_argument(f'--{prefix}num_folds_valid', type=int, default=None, help='if specified, only load data fold by fold')
        parser.add_argument(f'--{prefix}random_fold_train', type=str2bool, nargs='?', const=True, default=False, help='if specified, use random starting fold')
        parser.add_argument(f'--{prefix}init_fold_train', type=int, default=0, help='if specified, cycle through folds starting at specified fold')
        parser.add_argument(f'--{prefix}init_fold_valid', type=int, default=0, help='if specified, cycle through folds starting at specified fold')
        parser.add_argument(f'--{prefix}data_specs', type=str, default=None, help='if specified, string indicating specificities of the data')
        parser.add_argument(f'--{prefix}from_vid', type=str2bool, nargs='?', const=True, default=False, help='if specified, data is stores as video files, otherwise frames by frames')
        parser.add_argument(f'--{prefix}have_frames', type=str2bool, nargs='?', const=True, default=False, help='if specified, has not only video files but also frames')
        parser.add_argument(f'--{prefix}is_seq', type=str2bool, nargs='?', const=True, default=False, help='if specified, take into account sequential aspect')
        parser.add_argument(f'--{prefix}vid_len', type=int, default=16, help='number of frames in produced videos')
        parser.add_argument(f'--{prefix}p2p_len', type=int, default=None, help='number of frames in produced point-to-point videos')
        parser.add_argument(f'--{prefix}load_vid_len', type=int, default=None, help='load video with more frames to do random subsampling')
        parser.add_argument(f'--{prefix}max_vid_step', type=int, default=1000, help='max frames skipped in random subsampling')
        parser.add_argument(f'--{prefix}vid_skip', type=int, default=1, help='number of frames to skip between each clip')
        parser.add_argument(f'--{prefix}guide_size', type=int, default=None, help='number of channels of intermediate representation')
        parser.add_argument(f'--{prefix}categories', type=str, nargs="+", help='labels for the videos')
        parser.add_argument(f'--{prefix}load_data', type=str2bool, nargs='?', const=True, default=False, help='if specified, load data information from file')
        parser.add_argument(f'--{prefix}save_data', type=str2bool, nargs='?', const=True, default=False, help='if specified, save data information so that it does not have to be recomputed everytime')
        parser.add_argument(f'--{prefix}force_compute_metadata', type=str2bool, nargs='?', const=True, default=False, help='if specified, force re-computation of metadata after dataset update')
        parser.add_argument(f'--{prefix}shuffle_valid', type=str2bool, nargs='?', const=True, default=False, help='if specified, both training and validation set are shuffled')
        parser.add_argument(f'--{prefix}no_h_flip', type=str2bool, nargs='?', const=True, default=False, help='if specified, do not horizontally flip the images for data argumentation')
        parser.add_argument(f'--{prefix}no_v_flip', type=str2bool, nargs='?', const=True, default=False, help='if specified, do not vertically flip the images for data argumentation')
        parser.add_argument(f'--{prefix}resize_img', type=int, nargs="+", default=None, help='if specified, resize images to specified h,w once they are loaded')
        parser.add_argument(f'--{prefix}resize_center_crop_img', type=int, default=None, help='if specified, square crop images to specified size once they are loaded')
        parser.add_argument(f'--{prefix}original_size', type=int, nargs="+", default=None, help='if resize, specify original size')
        parser.add_argument(f'--{prefix}min_zoom', type=float, default=1., help='parameter for augmentation method consisting in zooming and cropping')
        parser.add_argument(f'--{prefix}max_zoom', type=float, default=1., help='parameter for augmentation method consisting in zooming and cropping')
        parser.add_argument(f'--{prefix}fixed_crop', type=int, nargs="+", default=None, help='if specified, apply a random crop of the given size')
        parser.add_argument(f'--{prefix}centered_crop', type=str2bool, nargs='?', const=True, default=False, help='if specified, cropped area is centered horizontally and vertically')
        parser.add_argument(f'--{prefix}fixed_top_centered_zoom', type=float, default=None, help='if specified, crop the image to the upper center part')
        parser.add_argument(f'--{prefix}elastic_transform', type=str2bool, nargs='?', const=True, default=False, help='if specified, apply random deformation to input image')
        parser.add_argument(f'--{prefix}num_workers', default=8, type=int, help='# threads for loading data')
        parser.add_argument(f'--{prefix}load_from_opt_file', type=str2bool, nargs='?', const=True, default=False, help='load options from checkpoints and use that as default')
        parser.add_argument(f'--{prefix}load_signature', type=str, default="", help='specifies experiment signature from which to load options')
        parser.add_argument(f'--{prefix}flow_bound', default=20, type=int, help='max number of pixels in displacement field')
        parser.add_argument(f'--{prefix}fps', default=10, type=int, help='frames per second')
        parser.add_argument(f'--{prefix}one_every_n', default=1, type=int, help='load one frame every n')
        parser.add_argument(f'--{prefix}load_state', type=str2bool, nargs='?', const=True, default=False, help='load state associated to frames')
        parser.add_argument(f'--{prefix}load_unc', type=str2bool, nargs='?', const=True, default=False, help='load frames from StyleGAN2 unconditional image generator')
        parser.add_argument(f'--{prefix}no_rgb_img_from_img', type=str2bool, nargs='?', const=True, default=False, help='do not load rgb even if specified')
        parser.add_argument(f'--{prefix}layout_size', default=None, type=int, help='do not load rgb even if specified')

        # for display and checkpointing
        parser.add_argument(f'--{prefix}log_freq', type=int, default=None, help='if specified, frequency at which logger is updated with images')
        parser.add_argument(f'--{prefix}log_fps', type=int, default=4, help='logs videos at specified speed in frames per second')
        parser.add_argument(f'--{prefix}save_freq', type=int, default=-1, help='frequency of saving models, if -1 don\'t save')
        parser.add_argument(f'--{prefix}save_latest_freq', type=int, default=5000, help='frequency of saving the latest model')
        parser.add_argument(f'--{prefix}save_path', type=str, default='./')

        # for loading
        parser.add_argument(f'--{prefix}cont_train', type=str2bool, nargs='?', const=True, default=False, help='continue training with model from which_iter')

        # for training
        parser.add_argument(f'--{prefix}n_iter', type=int, default=1000, help='number of training iterations')
        parser.add_argument(f'--{prefix}n_iter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument(f'--{prefix}iter_function', type=str, default="iter", help='(iter | cycle)')
        parser.add_argument(f'--{prefix}use_extra_dataset', type=str2bool, nargs='?', const=True, default=False, help="if specified, train on two datasets simultaneoulsy")

        # for evaluating online
        parser.add_argument(f'--{prefix}n_iter_eval', type=int, default=None, help='if specified, number of iterations between each evaluation phase')
        parser.add_argument(f'--{prefix}max_eval_batches', type=int, default=None, help='if specified, max number of eval batches to speed up evaluation')
        parser.add_argument(f'--{prefix}n_fvd', type=int, default=1024, help='number of videos to consider for Fréchet Video Distance')
        parser.add_argument(f'--{prefix}n_fid', type=int, default=1024, help='number of images to consider for Fréchet Inception Distance')

        # for evaluating offline
        parser.add_argument(f'--{prefix}model_to_evaluate', type=str, default="vid", help='(vid | two_stage_vid)')

        # for visualizing offline
        parser.add_argument(f'--{prefix}model_to_visualize', type=str, default="vid", help='(vid | two_stage_vid)')

        # for engine
        parser.add_argument(f'--{prefix}local_rank', type=int, default=0, help='process rank on node')

        # for generator
        parser.add_argument(f'--{prefix}rec_only', type=str2bool, nargs='?', const=True, default=False, help='if specified, only generate recovered video')
        parser.add_argument(f'--{prefix}step_by_step', type=str2bool, nargs='?', const=True, default=False, help='if specified, adapt predicted code from transformer to match actual synthesized frame')
        parser.add_argument(f'--{prefix}gen_from_img', type=str2bool, nargs='?', const=True, default=False, help='if specified, generate video from an image')
        parser.add_argument(f'--{prefix}keep_state', type=str2bool, nargs='?', const=True, default=False, help='if specified, generate video while keeping original state')
        parser.add_argument(f'--{prefix}custom_state', type=str2bool, nargs='?', const=True, default=False, help='if specified, generate video with custom state')
        parser.add_argument(f'--{prefix}layout', type=str2bool, nargs='?', const=True, default=False, help='if specified, use layouts as state')
        parser.add_argument(f'--{prefix}include_id', type=str2bool, nargs='?', const=True, default=False, help='if specified, include original video index in output name')
        parser.add_argument(f'--{prefix}down_size', type=int, nargs="+", default=None, help='if specified, decrease quality of input video to match target size')

        return parser

    def initialize_qvid_generator(self, parser):
        # for model
        parser.add_argument('--q_two_stage', type=str2bool, nargs='?', const=True, default=False, help='if specified, generate intermediate guides from which rgb frames are computed')
        parser.add_argument('--q_enc_model', type=str, default="taming", help='(taming|stylegan2)')
        parser.add_argument('--q_dec_model', type=str, default="stylegan2", help='(taming|stylegan2)')
        parser.add_argument('--q_use_ema', type=str2bool, nargs='?', const=True, default=False, help='if specified, store exponential moving average of weights')

        # for training
        parser.add_argument('--q_optimizer', type=str, default='adam')
        parser.add_argument('--q_beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--q_beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--q_weight_decay', type=float, default=0.0, help='weight decay term of adam')
        parser.add_argument('--q_lr', type=float, default=0.0000045, help='initial learning rate for adam')
        parser.add_argument('--q_decoder_only', type=str2bool, nargs='?', const=True, default=False, help='fine-tune decoder, freeze other parts of the model')
        parser.add_argument('--q_gan_loss', type=str, default="hinge", help='(original|hinge|wgan)')
        parser.add_argument('--q_is_continuous', type=str2bool, nargs='?', const=True, default=False, help='if specified, produce continuous tokens (do not quantize)')
        parser.add_argument('--q_use_enc', type=str2bool, nargs='?', const=True, default=False, help='when specified an encoder is used')
        parser.add_argument('--q_use_dec', type=str2bool, nargs='?', const=True, default=False, help='when specified a decoder is used')
        parser.add_argument('--q_use_di', type=str2bool, nargs='?', const=True, default=False, help='when specified an image discriminator is used')
        parser.add_argument('--q_use_di2', type=str2bool, nargs='?', const=True, default=False, help='when specified a second (unconditional) image discriminator is used')
        parser.add_argument('--q_use_dv', type=str2bool, nargs='?', const=True, default=False, help='when specified a video discriminator is used')
        parser.add_argument('--q_use_df', type=str2bool, nargs='?', const=True, default=False, help='when specified a feature discriminator between img and vid data is used')
        parser.add_argument('--q_use_categories', type=str2bool, nargs='?', const=True, default=False, help='when specified ground truth categories are used for training the discriminator')
        parser.add_argument('--q_use_vgg_img', type=str2bool, nargs='?', const=True, default=False, help='when specified vgg loss is used')
        parser.add_argument('--q_use_vgg_vid', type=str2bool, nargs='?', const=True, default=False, help='when specified vgg loss is used')
        parser.add_argument('--q_use_gan_feat_img', type=str2bool, nargs='?', const=True, default=False, help='when specified gan feat loss is used')
        parser.add_argument('--q_use_direct_recovery_img', type=str2bool, nargs='?', const=True, default=False, help='when specified recovery loss is used for img')
        parser.add_argument('--q_use_direct_recovery_vid', type=str2bool, nargs='?', const=True, default=False, help='when specified recovery loss is used for vid')
        parser.add_argument('--q_use_adaptive_lambda', type=str2bool, nargs='?', const=True, default=False, help='when specified the perceptual and adversarial losses are scaled such that they have similar gradients')
        parser.add_argument('--q_use_quant_loss_vid', type=str2bool, nargs='?', const=True, default=False, help='when specified quantization loss is used for vid')
        parser.add_argument('--q_use_entropy_img', type=str2bool, nargs='?', const=True, default=False, help='when specified entropy loss is used for img')
        parser.add_argument('--q_use_inter_rec_loss_img', type=str2bool, nargs='?', const=True, default=False, help='when specified inter recovery loss is used for img')
        parser.add_argument('--q_use_backwarp_consistency_img', type=str2bool, nargs='?', const=True, default=False, help='when specified wrap img with flow and compute consistency loss')
        parser.add_argument('--q_use_elastic_flow_recovery', type=str2bool, nargs='?', const=True, default=False, help='when specified compute regression loss on elastic image flow')
        parser.add_argument('--q_use_unc_gen', type=str2bool, nargs='?', const=True, default=False, help='when specified compute img discriminator loss without using context')
        parser.add_argument('--q_gan_start_iter', type=int, default=0, help='weight for quantization loss')
        parser.add_argument('--q_lambda_quant', type=float, default=1., help='weight for quantization loss')
        parser.add_argument('--q_lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--q_lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--q_lambda_gan', type=float, default=1.0, help='weight for gan loss')
        parser.add_argument('--q_lambda_di2', type=float, default=0.01, help='weight for second image discriminator')
        parser.add_argument('--q_lambda_flow_consistency', type=float, default=1., help='weight for flow consistency loss')
        parser.add_argument('--q_no_q_img', type=str2bool, nargs='?', const=True, default=False, help='no quantization loss for img')

        # for encoder
        parser.add_argument('--q_necf', type=int, default=128, help='# of enc filters in first conv layer')
        parser.add_argument('--q_necf_mult', type=int, nargs="+", default=[1, 1, 2, 2, 4], help='mult to obtain subsequent filters, downsampling by len(necf_mult)-1 times')
        parser.add_argument('--q_z_size', type=int, default=256, help='number of channels after encoding stage')
        parser.add_argument('--q_necr', type=int, default=2, help='# residual blocks for each resolution in encoder')

        # for decoder
        parser.add_argument('--q_ndcf', type=int, default=128, help='# of dec filters in first conv layer')
        parser.add_argument('--q_ndcf_mult', type=int, nargs="+", default=[1, 1, 2, 2, 4], help='mult to obtain subsequent filters, upsampling by len(ndcf_mult)-1 times')
        parser.add_argument('--q_d_size', type=int, default=3, help='number of channels after decoding stage')
        parser.add_argument('--q_to_mask', type=str2bool, nargs='?', const=True, default=False, help='decoder generates masks instead of rgb outputs')
        parser.add_argument('--q_ndcr', type=int, default=3, help='# residual blocks for each resolution in decoder')

        # for quantizer
        parser.add_argument('--q_z_num', type=int, default=256, help='number of embeddings for quantizing encoded features')
        parser.add_argument('--q_z_mult', type=int, default=1, help='number of cat embeddings per position')
        parser.add_argument('--q_z_shape', type=int, nargs="+", default=[16, 16], help='shape of channels after encoding')
        parser.add_argument('--q_use_q_anyway', type=str2bool, nargs='?', const=True, default=False, help='if continuous mode, but still use quantizer on encoded embedings')

        # for loading
        parser.add_argument('--q_load_path', type=str, default=None, help='load model from which_iter at specified folder')
        parser.add_argument('--q_which_iter', type=str, default=0, help='load the model from specified iteration, can be int, "latest" or "best"')
        parser.add_argument('--q_not_strict', type=str2bool, nargs='?', const=True, default=False, help='whether checkpoint exactly matches network architecture')
        parser.add_argument('--q_block_delta', type=int, default=None, help='helpful when loading checkpoints from a different resolution')

        # stylegan2
        parser.add_argument('--q_g_reg_every', type=int, default=None, help='interval for applying path length regularization')
        parser.add_argument('--q_d_reg_every', type=int, default=None, help='interval for applying r1 regularization')
        parser.add_argument('--q_vid_step_every', type=int, default=1, help='interval for applying a learning step on video data')
        parser.add_argument('--q_use_aug', type=str2bool, nargs='?', const=True, default=False, help='apply non leaking augmentation')
        parser.add_argument('--q_aug_p', type=float, default=0, help='probability of applying augmentation, 0 = use adaptive augmentation')
        parser.add_argument("--q_ada_target", type=float, default=0.6, help="target augmentation probability for adaptive augmentation")
        parser.add_argument("--q_ada_length", type=int, default=500 * 1000, help="target duraing to reach augmentation probability for adaptive augmentation")
        parser.add_argument("--q_lambda_r1", type=float, default=10, help="weight of the r1 regularization")
        parser.add_argument("--q_downsample_vdis_num", type=int, default=0, help="downsample video dis input num times")
        parser.add_argument("--q_downsample_dis_num", type=int, default=0, help="downsample image dis input num times")
        parser.add_argument("--q_stddev_group", type=int, default=4, help="groups for discriminator statistics")
        parser.add_argument('--q_n_consecutive_dis', type=int, default=1, help='run discriminator on n consecutive images')

        # skip
        parser.add_argument('--q_inter_p', type=float, default=0.5, help='prop of features to use in skip connections')
        parser.add_argument('--q_inter_drop_p', type=float, default=0.0, help='prop of inter features to drop')
        parser.add_argument('--q_use_inter', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--q_use_masked_flow', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--q_use_deformed_conv', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--q_use_tradeoff', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--q_no_corr', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--q_no_proj', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--q_is_pyramid', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--q_slide_inter', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--q_normalize_out', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--q_progressive_skip', type=int, default=None, help='progressive warm-up steps for skip auto-encoder')
        parser.add_argument('--q_skip_mode', type=str, default="enc", help='which intermediate features to use when propagating to future frames')
        parser.add_argument('--q_skip_context', type=int, nargs="+", default=[1], help='time delta of frames from which to propagate intermediate features')
        parser.add_argument('--q_keep_first', type=str2bool, nargs='?', const=True, default=False, help='keep first frame in context')
        parser.add_argument('--q_n_first', type=int, default=1, help='keep n first frame in context')
        parser.add_argument('--q_p2p_context', type=str2bool, nargs='?', const=True, default=False, help='train to reconstruct from start and end points')
        parser.add_argument('--q_skip_memory', type=int, default=1, help='number of time steps for which to store intermediate features')
        parser.add_argument('--q_skip_rgb', type=str2bool, nargs='?', const=True, default=False, help='construct rgb output by concatenating intermediate resolutions')
        parser.add_argument('--q_skip_tanh', type=str2bool, nargs='?', const=True, default=False, help='apply tanh on rbg output')

        # layout
        parser.add_argument('--q_use_layout', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--q_same_decoder_layout', type=str2bool, nargs='?', const=True, default=False)

        return parser

    def initialize_transformer(self, parser):
        # for model
        parser.add_argument('--x_z_num', type=int, default=256, help='number of tokens in vocabulary')
        parser.add_argument('--x_z_len', type=int, default=256, help='maximum sequence length of transformer')
        parser.add_argument('--x_num_blocks', type=int, default=16, help='number of unit blocks')
        parser.add_argument('--x_cond_len', type=int, default=256, help='conditioning length of transformer')
        parser.add_argument('--x_z_chunk', type=int, default=256, help='chunk size within sequence')
        parser.add_argument('--x_n_layer', type=int, default=24, help='number of transformer layers')
        parser.add_argument('--x_n_head', type=int, default=16, help='number of heads for multi-head attention')
        parser.add_argument('--x_n_embd', type=int, default=1024, help='dimension of token embeddings')
        parser.add_argument('--x_n_in', type=int, default=3, help='dimension of input for continuous transformer')
        parser.add_argument('--x_is_continuous', type=str2bool, nargs='?', const=True, default=False, help='if specified, use continuous tokens')
        parser.add_argument('--x_use_noise', type=str2bool, nargs='?', const=True, default=False, help='if specified, takes noise as input')
        parser.add_argument('--x_use_momentum', type=str2bool, nargs='?', const=True, default=False, help='if specified, penalize all')
        parser.add_argument('--x_lambda_momentum', type=float, default=0.01, help='weight for momentum')
        parser.add_argument('--x_n_proposals', type=int, default=1, help='in continuous case, number of proposals to predict')

        # for training
        parser.add_argument('--x_optimizer', type=str, default='adamw')
        parser.add_argument('--x_beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--x_beta2', type=float, default=0.95, help='momentum term of adam')
        parser.add_argument('--x_lr', type=float, default=0.0000045, help='initial learning rate for adam')
        parser.add_argument('--x_lr_warmup_iter', type=int, default=1, help='warmup iterations for learning rate')
        parser.add_argument('--x_lr_decay', type=str2bool, nargs='?', const=True, default=False, help='use cosine decay for learning rate')
        parser.add_argument('--x_lambda_nrec', type=float, default=1., help='weight for nearest embedding regression loss')
        parser.add_argument('--x_finetune_head', type=str2bool, nargs='?', const=True, default=False, help='only train head')
        parser.add_argument('--x_finetune_f', type=float, default=None, help='factor by which to lower lr of transformer blocks')

        # for loading
        parser.add_argument('--x_load_path', type=str, default=None, help='load model from which_iter at specified folder')
        parser.add_argument('--x_which_iter', type=str, default=0, help='load the model from specified iteration')
        parser.add_argument('--x_not_strict', type=str2bool, nargs='?', const=True, default=False, help='whether checkpoint exactly matches network architecture')
        parser.add_argument('--x_head_to_n', type=int, default=0, help='extend single head to multi hypotheses head')

        # for generating
        parser.add_argument('--x_sample', type=str2bool, nargs='?', const=True, default=False, help='using sampling when generating instead of top-1')
        parser.add_argument('--x_no_sample', type=str2bool, nargs='?', const=True, default=False, help='prevent sampling')
        parser.add_argument('--x_temperature', type=float, default=1.0, help='temperature when sampling')
        parser.add_argument('--x_top_k', type=int, default=None, help='if sepcified apply top-k when sampling')
        parser.add_argument('--x_beam_size', type=int, default=None, help='number of hypotheses for beam search')

        # for continuous
        parser.add_argument('--x_resid_noise', type=str2bool, nargs='?', const=True, default=False, help='use noise before activations in GPT blocks')
        parser.add_argument('--x_normalize_tgt', type=str2bool, nargs='?', const=True, default=False, help='l2-normalization of target embeddings')
        parser.add_argument('--x_normalize_pred', type=str2bool, nargs='?', const=True, default=False, help='l2-normalization of predicted embeddings')
        parser.add_argument('--x_continuous_loss', type=str, default='cosine', help='(vmf|cosine)')
        parser.add_argument('--x_epsilon_other', type=float, default=0.001, help='small update of badly initialized proposals')
        parser.add_argument('--x_knn', type=int, default=None, help='if sepcified compute loss on nearest neighbors')
        parser.add_argument('--x_knn_decay_iter', type=int, default=30000, help='decay iterations of nearest neighbors')

        # for decomposition
        parser.add_argument('--x_emb_mode', type=str, default=None, help='(spatio-temporal|temporal|None)')
        parser.add_argument('--x_z_shape', type=int, nargs="+", default=None, help='shape of channels after encoding, if None copy the value from qvid')

        # for point-to-point generation
        parser.add_argument('--x_p2p', type=str2bool, nargs='?', const=True, default=False, help='point-to-point generation mode')

        # for state-conditional generation
        parser.add_argument('--x_state', type=str2bool, nargs='?', const=True, default=False, help='state generation mode')
        parser.add_argument('--x_state_front', type=str2bool, nargs='?', const=True, default=False, help='the whole state sequence is treated as a video-level annotation')
        parser.add_argument('--x_state_num', type=int, default=None, help='number of embeddings for quantizing encoded features')
        parser.add_argument('--x_state_size', type=int, default=None, help='dimension of state')
        parser.add_argument('--x_sample_state', type=str2bool, nargs='?', const=True, default=False, help='using sampling when generating instead of top-1')
        parser.add_argument('--x_temperature_state', type=float, default=1.0, help='temperature when sampling')
        parser.add_argument('--x_top_k_state', type=int, default=None, help='if sepcified apply top-k when sampling')

        # for unconditional generation
        parser.add_argument('--x_use_start_token', type=str2bool, nargs='?', const=True, default=False, help='to launch the generation without prior information')

        # for class conditionned generation
        parser.add_argument('--x_cat', type=str2bool, nargs='?', const=True, default=False, help='class generation mode')

        # for sound conditionned generation
        parser.add_argument('--x_stft', type=str2bool, nargs='?', const=True, default=False, help='sound generation mode')

        # for blurred images conditionned synthesis
        parser.add_argument('--x_deblurring', type=str2bool, nargs='?', const=True, default=False, help='blurry image generation mode')
        parser.add_argument('--x_blur_sigma', type=int, default=10, help='blur image by specified sigma')

        return parser

    def initialize_state_estimator(self, parser):
        # for training
        parser.add_argument('--s_optimizer', type=str, default='adam')
        parser.add_argument('--s_beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--s_beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--s_weight_decay', type=float, default=0.0, help='weight decay term of adam')
        parser.add_argument('--s_lr', type=float, default=0.001, help='initial learning rate for adam')

        # for model
        parser.add_argument('--s_z_size', type=int, default=None, help='number of channels of input features')
        parser.add_argument('--s_z_shape', type=int, nargs="+", default=None, help='shape of channels after encoding')
        parser.add_argument('--s_state_hsize', type=int, default=128, help='dimension of hidden features')
        parser.add_argument('--s_state_size', type=int, default=0, help='dimension of state')
        parser.add_argument('--s_quantize_only', type=str2bool, nargs='?', const=True, default=False, help='do not use state estimator but only state quantizer')

        # for quantizer
        parser.add_argument('--s_state_num', type=int, default=0, help='number of embeddings for quantizing encoded features')

        # for loading
        parser.add_argument('--s_load_path', type=str, default=None, help='load model from which_iter at specified folder')
        parser.add_argument('--s_which_iter', type=str, default=0, help='load the model from specified iteration, can be int, "latest" or "best"')
        parser.add_argument('--s_not_strict', type=str2bool, nargs='?', const=True, default=False, help='whether checkpoint exactly matches network architecture')

        return parser

    def initialize_stft_ae(self, parser):
        # for training
        parser.add_argument('--a_optimizer', type=str, default='adam')
        parser.add_argument('--a_beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--a_beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--a_weight_decay', type=float, default=0.0, help='weight decay term of adam')
        parser.add_argument('--a_lr', type=float, default=0.001, help='initial learning rate for adam')

        # for model
        parser.add_argument('--a_stft_size', type=int, default=None, help='number of channels of latent features')
        parser.add_argument('--a_stft_shape', type=int, nargs="+", default=None, help='shape of channels after encoding')
        parser.add_argument('--a_stft_hsize', type=int, default=128, help='dimension of hidden features')

        # for quantizer
        parser.add_argument('--a_stft_num', type=int, default=None, help='number of embeddings for quantizing encoded features')

        # for loading
        parser.add_argument('--a_load_path', type=str, default=None, help='load model from which_iter at specified folder')
        parser.add_argument('--a_which_iter', type=str, default=0, help='load the model from specified iteration, can be int, "latest" or "best"')
        parser.add_argument('--a_not_strict', type=str2bool, nargs='?', const=True, default=False, help='whether checkpoint exactly matches network architecture')

        return parser

    def update_defaults(self, opt, parser):
        if getattr(opt, "x_z_shape") is None:
            parser.set_defaults(**{"x_z_shape": getattr(opt, "q_z_shape")})
        if getattr(opt, "x_state_num") is None:
            parser.set_defaults(**{"x_state_num": getattr(opt, "s_state_num")})
        if getattr(opt, "x_state_size") is None:
            parser.set_defaults(**{"x_state_size": getattr(opt, "s_state_size")})
        if getattr(opt, "s_z_shape") is None:
            parser.set_defaults(**{"s_z_shape": getattr(opt, "q_z_shape")})
        if getattr(opt, "s_z_size") is None:
            parser.set_defaults(**{"s_z_size": getattr(opt, "q_z_size")})
        for prefix in ["", "xb_"]:
            if getattr(opt, f"{prefix}dim") == -1:
                parser.set_defaults(**{f"{prefix}dim":getattr(opt, f"{prefix}max_dim")})
            if getattr(opt, f"{prefix}dataset") == "bairhd":
                parser.set_defaults(**{f"{prefix}dataroot": "datasets/bairhd"})
                parser.set_defaults(**{f"{prefix}true_ratio": 1})
                parser.set_defaults(**{f"{prefix}aspect_ratio": 1})
                parser.set_defaults(**{f"{prefix}true_dim": 256})
                parser.set_defaults(**{f"{prefix}categories": None})
                parser.set_defaults(**{f"{prefix}no_h_flip": True})
                parser.set_defaults(**{f"{prefix}no_v_flip": True})
                parser.set_defaults(**{f"{prefix}from_vid": False})
                parser.set_defaults(**{f"{prefix}fps": 4})
            if getattr(opt, f"{prefix}dataset") == "kinetics600":
                parser.set_defaults(**{f"{prefix}dataroot": "datasets/kinetics"})
                parser.set_defaults(**{f"{prefix}resize_center_crop_img": 256})
                parser.set_defaults(**{f"{prefix}true_ratio": 1})
                parser.set_defaults(**{f"{prefix}aspect_ratio": 1})
                parser.set_defaults(**{f"{prefix}true_dim": 256})
                parser.set_defaults(**{f"{prefix}categories": KINETICS600_CAT})
                parser.set_defaults(**{f"{prefix}no_h_flip": True})
                parser.set_defaults(**{f"{prefix}from_vid": True})
                parser.set_defaults(**{f"{prefix}imagenet_norm": True})
            if getattr(opt, f"{prefix}dataset") == "drums":
                parser.set_defaults(**{f"{prefix}dataroot": "datasets/drums"})
                parser.set_defaults(**{f"{prefix}true_ratio": 1})
                parser.set_defaults(**{f"{prefix}aspect_ratio": 1})
                parser.set_defaults(**{f"{prefix}true_dim": 96})
                parser.set_defaults(**{f"{prefix}categories": None})
                parser.set_defaults(**{f"{prefix}no_h_flip": True})
                parser.set_defaults(**{f"{prefix}from_vid": True})
                parser.set_defaults(**{f"{prefix}fps": 30})
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get options
        opt = parser.parse_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)
            # get options
            opt = parser.parse_args()

        # modify some defaults based on parser
        parser = self.update_defaults(opt, parser)
        opt = parser.parse_args()

        self.parser = parser
        return opt

    def print_options(self, opt, opt_type, opt_prefix=""):
        def dash_pad(s, length=50):
            num_dash = max(length - len(s) // 2, 0)
            return '-' * num_dash
        opt_str = opt_type + " Options"
        message = ''
        message += dash_pad(opt_str) + ' ' + opt_str + ' ' + dash_pad(opt_str) + '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(opt_prefix + k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        end_str = opt_type + " End"
        message += dash_pad(end_str) + ' ' + end_str + ' ' + dash_pad(end_str) + '\n'
        print(message)

    def option_file_path(self, opt, signature, makedir=False):
        expr_dir = os.path.join(opt.save_path, "checkpoints", signature)
        if makedir:
            utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt, signature):
        file_name = self.option_file_path(opt, signature, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, opt.load_signature, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def split_options(self, opt):
        base_opt = Namespace()
        extra_base_opt = Namespace()
        qvid_generator_opt = Namespace()
        transformer_opt = Namespace()
        state_estimator_opt = Namespace()
        stft_ae_opt = Namespace()
        for k, v in sorted(vars(opt).items()):
            if k.startswith("xb_"):
                setattr(extra_base_opt, k[3:], v)
            elif k.startswith("q_"):
                setattr(qvid_generator_opt, k[2:], v)
            elif k.startswith("x_"):
                setattr(transformer_opt, k[2:], v)
            elif k.startswith("s_"):
                setattr(state_estimator_opt, k[2:], v)
            elif k.startswith("a_"):
                setattr(stft_ae_opt, k[2:], v)
            else:
                setattr(base_opt, k, v)
        return base_opt, extra_base_opt, qvid_generator_opt, transformer_opt, state_estimator_opt, stft_ae_opt

    def copy_options(self, target_options, source_options, new_only=False):
        for k, v in sorted(vars(source_options).items()):
            if not (new_only and k in target_options):
                setattr(target_options, k, v)

    def process_base(self, base_opt, signature):
        # set gpu ids
        str_ids = base_opt.gpu_ids.split(',')
        base_opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                base_opt.gpu_ids.append(id)

        # set hr dim
        base_opt.hr_dim = base_opt.hr_dim if base_opt.hr_dim is not None else base_opt.true_dim

        # set additional paths
        base_opt.checkpoint_path = os.path.join(base_opt.save_path, "checkpoints", signature)
        base_opt.log_path = os.path.join(base_opt.save_path, "logs", signature)
        base_opt.result_path = os.path.join(base_opt.save_path, "results", signature)

        assert (base_opt.max_dim & (base_opt.max_dim - 1)) == 0, f"Max dim {base_opt.max_dim} must be power of two."

        # set width size
        if base_opt.fixed_crop is None:
            base_opt.width_size = int(base_opt.dim * base_opt.aspect_ratio)
            base_opt.height_size = int(base_opt.width_size / base_opt.aspect_ratio)
        else:
            base_opt.height_size, base_opt.width_size = base_opt.fixed_crop

        # set resize factor
        if base_opt.resize_img is not None:
            print("Resize img to", base_opt.resize_img)
            assert base_opt.original_size is not None
            base_opt.resize_factor_h = base_opt.true_dim / base_opt.original_size[0]
            base_opt.resize_factor_w = base_opt.true_dim * base_opt.true_ratio / base_opt.original_size[1]
        else:
            base_opt.resize_factor_h = 1
            base_opt.resize_factor_w = 1

        # set signature
        base_opt.signature = signature

    def parse(self, load_qvid_generator=False, load_transformer=False, load_extra_base=False, load_state_estimator=False, load_stft_ae=False, save=False):
        opt = self.gather_options()
        signature = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "-" + opt.name

        base_opt, extra_base_opt, qvid_generator_opt, transformer_opt, state_estimator_opt, stft_ae_opt = self.split_options(opt)

        if base_opt.local_rank == 0:
            if save:
                self.save_options(opt, signature)
            self.print_options(base_opt, "Base")
            if load_extra_base and base_opt.use_extra_dataset:
                self.print_options(extra_base_opt, "Extra Base", "xb_")
            if load_qvid_generator:
                self.print_options(qvid_generator_opt, "QVideo Generator", "q_")
            if load_transformer:
                self.print_options(transformer_opt, "Transformer", "x_")
            if load_state_estimator:
                self.print_options(state_estimator_opt, "State Estimator", "s_")
            if load_stft_ae:
                self.print_options(stft_ae_opt, "Stft Auto Encoder", "a_")

        self.process_base(base_opt, signature)
        if load_extra_base and base_opt.use_extra_dataset:
            self.process_base(extra_base_opt, signature)

        self.copy_options(qvid_generator_opt, base_opt)
        self.copy_options(transformer_opt, base_opt)
        self.copy_options(state_estimator_opt, base_opt)
        self.copy_options(stft_ae_opt, base_opt)

        self.base_opt = base_opt
        self.extra_base_opt = extra_base_opt if load_extra_base else None
        self.qvid_generator_opt = qvid_generator_opt if load_qvid_generator else None
        self.transformer_opt = transformer_opt if load_transformer else None
        self.state_estimator_opt = state_estimator_opt if load_state_estimator else None
        self.stft_ae_opt = stft_ae_opt if load_stft_ae else None

        self.opt = {"base": self.base_opt,
                    "extra_base":  self.extra_base_opt,
                    "qvid_generator": self.qvid_generator_opt,
                    "transformer": self.transformer_opt,
                    "state_estimator": self.state_estimator_opt,
                    "stft_ae": self.stft_ae_opt}

        return self.opt