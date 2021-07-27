import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from ..models.gan import StyleGAN2Discriminator, StyleGAN2VidDiscriminator, FeatureDiscriminator
from ..models.skip_autoencoder import SkipGANDecoder, SkipGANEncoder
from ..modules.quantize import VectorQuantizer
from ..modules.gan_loss import get_gan_loss
from ..modules.perceptual import VGGLoss
from ..modules.non_leaking import AdaptiveAugment, augment
from tools.utils import to_cuda, DummyOpt
from models import load_network, save_network, print_network

class QVidModel(torch.nn.Module):
    def __init__(self, opt, is_train=True, is_main=True, logger=None):
        super().__init__()
        self.opt = opt
        self.is_main = is_main

        self.initialize_networks(is_train)

        if is_train:
            self.opt_g, self.opt_d = self.create_optimizers(self.opt)
            gan_loss = get_gan_loss(self.opt)
            if self.opt.use_di:
                self.gan_loss_img = gan_loss(self.net_di)
            if self.opt.use_dv:
                self.gan_loss_vid = gan_loss(self.net_dv)
            if self.opt.use_df:
                self.gan_loss_feat = gan_loss(self.net_df)
            if self.opt.use_vgg_img:
                self.vgg_loss_img = VGGLoss(self.opt.gpu_ids)
            if self.opt.use_gan_feat_img:
                self.feat_loss_img = torch.nn.L1Loss()
            self.ada_aug_p = self.opt.aug_p if self.opt.aug_p > 0 else 0.0
            if self.opt.use_aug and self.opt.aug_p == 0:
                self.ada_aug = AdaptiveAugment(self.opt.ada_target, self.opt.ada_length, 8, device="cuda")
            if self.opt.use_layout:
                self.layout_loss = nn.CrossEntropyLoss()

        self.logger = logger if self.is_main else None


    def forward(self, data, fake_data={}, mode='', log=False, suffix="", cond_frames=None, global_iter=None):
        _, real_img, real_layout, real_flow_img, real_mask_img, _, real_edge, real_vid, _, code, state_code, inter, interl, cond_inter = self.preprocess_input(data)
        z, fake_img, _, _, _, fake_soft_mask_img, _, fake_vid, fake_unc_vid, _, _, _, _, _ = self.preprocess_input(fake_data, is_fake=True)

        if mode == 'img_to_img_generator':
            g_loss, fake_data = self.compute_img_to_img_generator_loss(real_img, real_layout, real_flow_img, real_mask_img, log, global_iter)
            return g_loss, fake_data

        if mode == 'eval_img_to_img_generator':
            rec_loss = self.compute_eval_img_to_img_generator_loss(real_img, log, global_iter)
            return rec_loss

        elif mode == 'vid_to_vid_generator':
            g_loss, fake_data = self.compute_vid_to_vid_generator_loss(real_vid, real_layout, log, global_iter)
            return g_loss, fake_data

        elif mode == 'img_discriminator':
            d_loss = self.compute_img_discriminator_loss(z, real_img, fake_img, log, global_iter)
            return d_loss

        elif mode == 'img_discriminator_reg':
            r_loss = self.compute_img_discriminator_regularization_loss(z, real_img, log, global_iter)
            return r_loss

        elif mode == 'vid_discriminator_reg':
            r_loss = self.compute_vid_discriminator_regularization_loss(z, real_vid, log, global_iter)
            return r_loss

        elif mode == 'vid_discriminator':
            d_loss = self.compute_vid_discriminator_loss(z, real_vid, fake_vid, fake_unc_vid, log, global_iter)
            return d_loss

        elif mode == 'img_encoder':
            encoded_data = self.encode(real_img, real_layout, "img", log, suffix, global_iter)
            return encoded_data

        elif mode == 'vid_encoder':
            encoded_data = self.encode(real_vid, real_layout, "vid", log, suffix, global_iter)
            return encoded_data

        elif mode == 'img_decoder':
            decoded_data = self.decode(code, state_code, inter, interl, cond_inter, "img", log, suffix, None, global_iter)
            return decoded_data

        elif mode == 'vid_decoder':
            decoded_data = self.decode(code, state_code, inter, interl, cond_inter, "vid", log, suffix, cond_frames, global_iter)
            return decoded_data

        elif mode == 'vid_step_decoder':
            decoded_data = self.vid_step_decode(code, inter, cond_inter)
            return decoded_data

        else:
            raise ValueError(f"mode '{mode}' is invalid")


    def preprocess_input(self, data, is_fake=False):
        data["z"] = to_cuda(data, "z")
        data["img"] = to_cuda(data, "img")
        data["layout"] = to_cuda(data, "layout")
        data["flow_img"] = to_cuda(data, "flow_img")
        data["mask_img"] = to_cuda(data, "mask_img")
        data["soft_mask_img"] = to_cuda(data, "soft_mask_img")
        data["edge_img"] = to_cuda(data, "edge_img")
        data["vid"] = to_cuda(data, "vid")
        data["unc_vid"] = to_cuda(data, "unc_vid")
        data["code"] = to_cuda(data, "code")
        data["state_code"] = to_cuda(data, "state_code")
        data["inter"] = to_cuda(data, "inter")
        data["interl"] = to_cuda(data, "interl")
        data["cond_inter"] = to_cuda(data, "cond_inter")
        if is_fake:
            data["z"] = data["z"].detach()
            data["img"] = data["img"].detach()
            data["vid"] = data["vid"].detach()
            data["unc_vid"] = data["unc_vid"].detach()
            data["soft_mask_img"] = data["soft_mask_img"].detach()
        return data["z"], data["img"], data["layout"], data["flow_img"], data["mask_img"], data["soft_mask_img"], data["edge_img"], data["vid"], data["unc_vid"], data["code"], data["state_code"], data["inter"], data["interl"], data["cond_inter"]


    def initialize_networks(self, is_train):
        self.net_di = StyleGAN2Discriminator(self.opt).cuda() if is_train and self.opt.use_di else None
        self.net_dv = StyleGAN2VidDiscriminator(self.opt).cuda() if is_train and self.opt.use_dv else None
        self.net_df = FeatureDiscriminator(self.opt).cuda() if is_train and self.opt.use_df else None

        self.net_e = None
        self.net_el = None
        if self.opt.use_enc:
            if self.opt.enc_model == "skipgan":
                self.net_e = SkipGANEncoder(self.opt).cuda()
                if self.opt.use_layout:
                    self.net_el = SkipGANEncoder(self.opt, mode="layout").cuda()
            else:
                raise ValueError

        self.net_q = None
        self.net_ql = None
        if not self.opt.is_continuous or self.opt.use_q_anyway:
            self.net_q = VectorQuantizer(self.opt.z_num, self.opt.z_size, beta=0.25, mult=self.opt.z_mult, normalize=self.opt.normalize_out).cuda()
            if self.opt.use_layout:
                self.net_ql = VectorQuantizer(self.opt.z_num, self.opt.z_size, beta=0.25, mult=self.opt.z_mult, normalize=self.opt.normalize_out).cuda()

        self.net_g = None
        self.net_gl = None
        if self.opt.use_dec:
            if self.opt.dec_model == "skipgan":
                if self.opt.use_layout:
                    if self.opt.same_decoder_layout:
                        self.net_g = SkipGANDecoder(self.opt, mode="both").cuda()
                    else:
                        self.net_g = SkipGANDecoder(self.opt).cuda()
                        self.net_gl = SkipGANDecoder(self.opt, mode="layout").cuda()
                else:
                    self.net_g = SkipGANDecoder(self.opt).cuda()
            else:
                raise ValueError

        self.net_e_ema = None
        self.net_q_ema = None
        self.net_g_ema = None
        self.net_el_ema = None
        self.net_ql_ema = None
        self.net_gl_ema = None
        if self.opt.use_ema and is_train:
            self.net_e_ema = SkipGANEncoder(self.opt).cuda().eval() if self.net_e is not None else None
            self.net_q_ema = VectorQuantizer(self.opt.z_num, self.opt.z_size, beta=0.25, mult=self.opt.z_mult, normalize=self.opt.normalize_out).cuda().eval() if self.net_q is not None else None
            self.net_g_ema = SkipGANDecoder(self.opt, mode=self.net_g.mode).cuda().eval() if self.net_g is not None else None
            self.net_el_ema = SkipGANEncoder(self.opt, mode="layout").cuda().eval() if self.net_el is not None else None
            self.net_ql_ema = VectorQuantizer(self.opt.z_num, self.opt.z_size, beta=0.25, mult=self.opt.z_mult, normalize=self.opt.normalize_out).cuda().eval() if self.net_ql is not None else None
            self.net_gl_ema = SkipGANDecoder(self.opt, mode="layout").cuda().eval() if self.net_gl is not None else None

        if self.is_main:
            load_ema = not is_train and self.opt.use_ema
            bd = self.opt.block_delta
            self.net_g = load_network(self.net_g, "qvid_g_ema", self.opt, block_delta=bd) if load_ema else load_network(self.net_g, "qvid_g", self.opt, block_delta=bd)
            self.net_e = load_network(self.net_e, "qvid_e_ema", self.opt) if load_ema else load_network(self.net_e, "qvid_e", self.opt)
            self.net_q = load_network(self.net_q, "qvid_q_ema", self.opt) if load_ema else load_network(self.net_q, "qvid_q", self.opt, required=False)
            self.net_gl = load_network(self.net_gl, "qvid_gl_ema", self.opt, block_delta=bd, required=False) if load_ema else load_network(self.net_gl, "qvid_gl", self.opt, block_delta=bd, required=False)
            self.net_el = load_network(self.net_el, "qvid_el_ema", self.opt, required=False) if load_ema else load_network(self.net_el, "qvid_el", self.opt, required=False)
            self.net_ql = load_network(self.net_ql, "qvid_ql_ema", self.opt) if load_ema else load_network(self.net_ql, "qvid_ql", self.opt, required=False)
            self.net_di = load_network(self.net_di, "qvid_di", self.opt, required=False, block_delta=bd)
            self.net_dv = load_network(self.net_dv, "qvid_dv", self.opt, required=False, block_delta=bd)
            self.net_df = load_network(self.net_df, "qvid_df", self.opt, required=False, block_delta=bd)
            if True:
                print_network(self.net_e)
                print_network(self.net_q)
                print_network(self.net_g)
                print_network(self.net_el)
                print_network(self.net_ql)
                print_network(self.net_gl)
                print_network(self.net_di)
                print_network(self.net_dv)
                print_network(self.net_df)
            if self.opt.use_ema and is_train:
                self.accumulate(0)
                self.net_g_ema = load_network(self.net_g_ema, "qvid_g_ema", self.opt, required=False)
                self.net_e_ema = load_network(self.net_e_ema, "qvid_e_ema", self.opt, required=False)
                self.net_q_ema = load_network(self.net_q_ema, "qvid_q_ema", self.opt, required=False)
                self.net_gl_ema = load_network(self.net_gl_ema, "qvid_gl_ema", self.opt, required=False)
                self.net_el_ema = load_network(self.net_el_ema, "qvid_el_ema", self.opt, required=False)
                self.net_ql_ema = load_network(self.net_ql_ema, "qvid_ql_ema", self.opt, required=False)


    def save_model(self, global_iter, latest=False, best=False):
        save_network(self.net_e, "qvid_e", global_iter, self.opt, latest, best)
        save_network(self.net_q, "qvid_q", global_iter, self.opt, latest, best)
        save_network(self.net_g, "qvid_g", global_iter, self.opt, latest, best)
        save_network(self.net_el, "qvid_el", global_iter, self.opt, latest, best)
        save_network(self.net_ql, "qvid_ql", global_iter, self.opt, latest, best)
        save_network(self.net_gl, "qvid_gl", global_iter, self.opt, latest, best)
        save_network(self.net_di, "qvid_di", global_iter, self.opt, latest, best)
        save_network(self.net_dv, "qvid_dv", global_iter, self.opt, latest, best)
        save_network(self.net_df, "qvid_df", global_iter, self.opt, latest, best)
        save_network(self.net_e_ema, "qvid_e_ema", global_iter, self.opt, latest, best)
        save_network(self.net_q_ema, "qvid_q_ema", global_iter, self.opt, latest, best)
        save_network(self.net_g_ema, "qvid_g_ema", global_iter, self.opt, latest, best)
        save_network(self.net_el_ema, "qvid_el_ema", global_iter, self.opt, latest, best)
        save_network(self.net_ql_ema, "qvid_ql_ema", global_iter, self.opt, latest, best)
        save_network(self.net_gl_ema, "qvid_gl_ema", global_iter, self.opt, latest, best)


    def create_optimizers(self, opt):
        if opt.optimizer == "adam":
            g_params, d_params = [], []
            if not self.opt.decoder_only:
                g_params += list(self.net_e.parameters()) if self.net_e is not None else []
                g_params += list(self.net_el.parameters()) if self.net_el is not None else []
                g_params += list(self.net_q.parameters()) if self.net_q is not None else []
                g_params += list(self.net_ql.parameters()) if self.net_ql is not None else []
            g_params += list(self.net_g.parameters()) if self.net_g is not None else []
            g_params += list(self.net_gl.parameters()) if self.net_gl is not None else []
            d_params += list(self.net_di.parameters()) if self.net_di is not None else []
            d_params += list(self.net_dv.parameters()) if self.net_dv is not None else []
            d_params += list(self.net_df.parameters()) if self.net_df is not None else []
            g_reg_ratio = opt.g_reg_every / (opt.g_reg_every + 1) if opt.g_reg_every is not None else 1
            d_reg_ratio = opt.d_reg_every / (opt.d_reg_every + 1) if opt.d_reg_every is not None else 1
            opt_g = torch.optim.Adam(g_params, lr=opt.lr * g_reg_ratio, betas=(opt.beta1 ** g_reg_ratio, opt.beta2 ** g_reg_ratio), weight_decay=opt.weight_decay)
            if d_params:
                opt_d = torch.optim.Adam(d_params, lr=opt.lr * d_reg_ratio, betas=(opt.beta1 ** d_reg_ratio, opt.beta2 ** d_reg_ratio), weight_decay=opt.weight_decay)
            else:
                opt_d = DummyOpt()
        else:
            raise NotImplementedError
        return opt_g, opt_d


    def compute_img_to_img_generator_loss(self, real_img, real_layout, real_flow_img, real_mask_img, log, global_iter):
        loss = torch.tensor(0., requires_grad=True).cuda()

        # encode image
        z, inter_enc = self.net_e(real_img)

        # encode layout
        if self.opt.use_layout:
            real_soft_layout = torch.zeros(real_layout.size(0), self.opt.layout_size, *real_layout.shape[-2:]).cuda().scatter_(1, real_layout, 1.0)
            zl, inter_encl = self.net_el(real_soft_layout)

        # quantize image
        quant_loss = None
        if self.opt.is_continuous:
            z_q = z # no quantization
        else:
            z_q, quant_loss, _ = self.net_q(z)
            quant_loss = quant_loss * self.opt.lambda_quant
            if not self.opt.no_q_img:
                loss += quant_loss

        # quantize layout
        layout_quant_loss = None
        if self.opt.use_layout:
            if self.opt.is_continuous:
                zl_q = zl  # no quantization
            else:
                zl_q, layout_quant_loss, _ = self.net_ql(zl)
                layout_quant_loss = layout_quant_loss * self.opt.lambda_quant
                if not self.opt.no_q_img:
                    loss += layout_quant_loss

        # shuffle intermediate encoded features for frames from the same sequence
        # such as to learn to reconstruct frames from neighbouring frames
        if self.opt.slide_inter:
            assert self.opt.n_consecutive_img > 1
            n = self.opt.n_consecutive_img
            indices = list(range(1, n)) + [0]
            tot = n + (1 if self.opt.load_elastic_view else 0)
            indices = indices + [0] if self.opt.load_elastic_view else indices
            inter_tgt = [f.view(f.size(0) // tot, tot, *f.shape[1:])[:, indices].contiguous().view_as(f) for f in inter_enc]
            if self.opt.use_layout:
                inter_tgtl = [f.view(f.size(0) // tot, tot, *f.shape[1:])[:, indices].contiguous().view_as(f) for f in inter_encl]
        else:
            if self.opt.load_elastic_view:
                indices = [0, 0]
                inter_tgt = [f.view(f.size(0) // 2, 2, *f.shape[1:])[:, indices].contiguous().view_as(f) for f in inter_enc]
                if self.opt.use_layout:
                    inter_tgtl = [f.view(f.size(0) // 2, 2, *f.shape[1:])[:, indices].contiguous().view_as(f) for f in inter_encl]
            else:
                inter_tgt = inter_enc
                if self.opt.use_layout:
                    inter_tgtl = inter_encl

        # we apply an elastic transformation to the original image
        # we then corrupt the original image by greying out some parts
        # the objective is to make reconstruction more challenging
        # to tackle both pure synthesis and re-use of past information
        corr_img = torch.tensor([])
        corr_layout = torch.tensor([])
        if self.opt.elastic_corruption:
            n = self.opt.n_consecutive_img
            bs = real_img.size(0)
            no_corr_idx = [i for i in range(bs) if i % (n + 1) != 0]
            corr_idx = [i for i in range(bs) if i % (n + 1) == 0]
            z_q = z_q[no_corr_idx]
            inter_tgt = [f[no_corr_idx] for f in inter_tgt]
            no_corr_img = real_img[no_corr_idx]
            corr_img = real_img[corr_idx]
            real_img = no_corr_img
            if self.opt.use_layout:
                zl_q = zl_q[no_corr_idx]
                inter_tgtl = [f[no_corr_idx] for f in inter_tgtl]
                no_corr_layout = real_layout[no_corr_idx]
                corr_layout = real_layout[corr_idx]
                real_layout = no_corr_layout

        # merge intermediate features from image and layout
        # half of intermediate features are occupied by image features and half by layout features
        if self.opt.use_layout and self.opt.same_decoder_layout:
            for i in range(len(inter_tgt)):
                half_size = inter_tgt[i].size(1) // 2
                inter_tgt[i] = torch.cat([inter_tgt[i][:, :half_size], inter_tgtl[i][:, half_size:]], dim=1)
            z_q = torch.cat([z_q, zl_q], dim=1)

        # decode
        fake_img, fake_layout, inter_flows, inter_occs, inter_dec = self.net_g(z_q, [inter_tgt], drop_p=self.opt.inter_drop_p, return_all=True)
        if self.opt.use_layout and not self.opt.same_decoder_layout:
            _, fake_layout = self.net_gl(zl_q, [inter_tgtl], drop_p=self.opt.inter_drop_p)
        flows = inter_flows
        if len(inter_occs) > 0:
            occ_mask = F.sigmoid(inter_occs[-1])
        else:
            occ_mask = torch.tensor([])

        layout_ce_loss = None
        if self.opt.use_layout:
            layout_ce_loss = self.layout_loss(fake_layout, real_layout.squeeze(1))
            loss += layout_ce_loss

        mask_rec_loss = None
        if self.opt.elastic_corruption:
            assert self.opt.load_elastic_view
            assert self.opt.inter_drop_p == 0
            bs = fake_img.size(0)
            n = self.opt.n_consecutive_img - 1
            elastic_idx = [i * (n + 1) + n for i in range(bs // (n + 1))]
            mask_rec_loss = F.mse_loss(occ_mask[elastic_idx][real_mask_img], torch.tensor([1.]).cuda())
            loss += mask_rec_loss

        inter_rec_loss = None
        if self.opt.use_inter_rec_loss_img:
            inter_rec_loss = torch.tensor(0., requires_grad=True).cuda()
            for i in range(len(inter_enc)):
                inter_rec_loss += F.mse_loss(inter_enc[i], inter_dec[-1-i])
            loss += inter_rec_loss

        elastic_flow_rec_loss = None
        if self.opt.use_elastic_flow_recovery:
            elastic_flow_rec_loss = 0
            assert self.opt.load_elastic_view
            assert self.opt.inter_drop_p == 0
            bs = fake_img.size(0)
            n = self.opt.n_consecutive_img
            n = n - 1 if self.opt.elastic_corruption else n
            elastic_idx = [i * (n + 1) + n for i in range(bs // (n + 1))]
            for fake_flow in flows:
                elastic_flow = fake_flow[elastic_idx]
                real_flow = F.interpolate(real_flow_img / self.net_g.last_flow_mult, size=elastic_flow.shape[-2:], mode='bilinear')
                if self.opt.elastic_corruption:
                    no_occ_mask = F.interpolate(real_mask_img.float(), size=elastic_flow.shape[-2:], mode='bilinear') < 0.5
                    no_occ_mask = no_occ_mask.repeat(1, 2, 1, 1)
                    elastic_flow_rec_loss += F.mse_loss(elastic_flow[no_occ_mask], real_flow[no_occ_mask])
                else:
                    elastic_flow_rec_loss += F.mse_loss(elastic_flow, real_flow)
            loss += elastic_flow_rec_loss
        flow = flows[-1] * self.net_g.last_flow_mult if len(flows) > 0 else torch.tensor([])

        backwarp_consistency_loss = None
        if self.opt.use_backwarp_consistency_img:
            assert self.opt.slide_inter
            n = self.opt.n_consecutive_img
            indices = list(range(1, n)) + [0]
            r = real_img
            r = r.view(r.size(0) // n, n, *r.shape[1:])[:, indices].contiguous().view_as(r)
            warped_real_img = self.net_g.backwarp_img(input=r, flow=flow)
            occ_sum = (1 - occ_mask).view(occ_mask.size(0), -1).sum(dim=1).view(occ_mask.size(0), 1, 1, 1)
            backwarp_consistency_loss = ((fake_img - warped_real_img) ** 2 * (1 - occ_mask) / occ_sum).mean()
            loss += backwarp_consistency_loss

        flow = flow / flow.size(2) if flow.size(0) > 0 else flow
        real_flow_img = real_flow_img / real_flow_img.size(2) if real_flow_img.size(0) > 0 else real_flow_img

        # direct recovery
        rec_loss = torch.mean(torch.abs(real_img - fake_img))
        if self.opt.use_direct_recovery_img:
            loss += rec_loss

        # vgg features matching
        vgg_loss = None
        if self.opt.use_vgg_img:
            vgg_loss = self.vgg_loss_img(fake_img, real_img) * self.opt.lambda_vgg
            loss += vgg_loss

        # adversarial
        gen_loss = None
        if self.opt.use_di:
            img_for_di = augment(fake_img, self.ada_aug_p)[0] if self.opt.use_aug else fake_img
            score_fake = self.net_di(img_for_di)["score"]
            gen_loss = self.gan_loss_img.generator_loss_logits(score_fake).sum() * self.opt.lambda_gan
            loss += gen_loss

        # adversairal
        gen_feat_loss = None
        if self.opt.use_df:
            score_fake = self.net_df(z_q)["score"]
            gen_feat_loss = self.gan_loss_feat.generator_loss_logits(score_fake).sum()
            loss += gen_feat_loss

        if self.logger:
            # log scalars every step
            self.logger.log_scalar("qvid_generator/gen_img", gen_loss, global_iter)
            self.logger.log_scalar("qvid_generator/layout_img", layout_ce_loss, global_iter)
            self.logger.log_scalar("qvid_generator/gen_feat_fake", gen_feat_loss, global_iter)
            self.logger.log_scalar("qvid_generator/quant_img", quant_loss, global_iter)
            self.logger.log_scalar("qvid_generator/layout_quant_img", layout_quant_loss, global_iter)
            self.logger.log_scalar("qvid_generator/rec_img", rec_loss, global_iter)
            self.logger.log_scalar("qvid_generator/vgg_img", vgg_loss, global_iter)
            self.logger.log_scalar("qvid_generator/mask_rec_img", mask_rec_loss, global_iter)
            self.logger.log_scalar("qvid_generator/inter_rec_img", inter_rec_loss, global_iter)
            self.logger.log_scalar("qvid_generator/elastic_flow_rec_img", elastic_flow_rec_loss, global_iter)
            self.logger.log_scalar("qvid_generator/backwarp_consistency_img", backwarp_consistency_loss, global_iter)
            # log images every few steps
            if log:
                self.logger.log_img("qvid_generator/corr_img", corr_img.cpu()[:16], 4, global_iter, normalize=True, span=(-1, 1))
                self.logger.log_img("qvid_generator/fake_img", fake_img[:16].float().cpu(), 4, global_iter, normalize=True, span=(-1, 1))
                self.logger.log_img("qvid_generator/mask", real_mask_img[:16].float().cpu(), 4, global_iter)
                self.logger.log_img("qvid_generator/occ_mask", occ_mask[:16].float().cpu(), 4, global_iter)
                self.logger.log_flow("qvid_generator/flow", flow[:16].float().cpu(), 4, global_iter)
                self.logger.log_flow("qvid_generator/real_flow", real_flow_img[:16].cpu(), 4, global_iter)
                corr_layout = corr_layout.squeeze(1) if corr_layout.size(0) > 0 else corr_layout
                real_layout = real_layout.squeeze(1) if real_layout.size(0) > 0 else real_layout
                self.logger.log_seg("qvid_generator/corr_layout", corr_layout[:16].cpu(), self.opt.layout_size, 4, global_iter)
                self.logger.log_seg("qvid_generator/real_layout", real_layout[:16].cpu(), self.opt.layout_size, 4, global_iter)
                self.logger.log_seg("qvid_generator/fake_layout", fake_layout[:16].cpu(), self.opt.layout_size, 4, global_iter)
        return loss, {"img": fake_img, "z": z_q}


    @torch.no_grad()
    def compute_eval_img_to_img_generator_loss(self, real_img, log, global_iter):
        # encode
        z, inter_enc = self.net_e(real_img)

        # quantize
        if self.opt.is_continuous:
            z_q = z # no quantization
        else:
            z_q, _, _ = self.net_q(z)

        # decode
        fake_img, _ = self.net_g(z_q, [inter_enc])

        # direct recovery
        loss = torch.mean(torch.abs(real_img - fake_img))

        if self.logger and log:
            self.logger.log_img("qvid_generator/eval_fake_img", fake_img[:16].float().cpu(), 4, global_iter, normalize=True, span=(-1, 1))
            self.logger.log_img("qvid_generator/eval_real_img", real_img[:16].float().cpu(), 4, global_iter, normalize=True, span=(-1, 1))

        return loss


    def compute_vid_to_vid_generator_loss(self, real_vid, real_layout, log, global_iter):
        loss = torch.tensor(0., requires_grad=True).cuda()

        # encode image
        z, inter_enc = self.net_e(real_vid)

        # encode layout
        if self.opt.use_layout and self.opt.same_decoder_layout:
            real_soft_layout = torch.zeros(*real_layout.shape[:2], self.opt.layout_size, *real_layout.shape[-2:]).cuda().scatter_(2, real_layout.unsqueeze(2), 1.0)
            zl, inter_encl = self.net_el(real_soft_layout)

        # quantize image
        quant_loss = None
        if self.opt.is_continuous:
            z_q = z  # no quantization
        else:
            z_q, quant_loss, _ = self.net_q(z)
            quant_loss = quant_loss * self.opt.lambda_quant
            loss += quant_loss

        # quantize layout
        layout_quant_loss = None
        if self.opt.use_layout and self.opt.same_decoder_layout:
            if self.opt.is_continuous:
                zl_q = zl  # no quantization
            else:
                zl_q, layout_quant_loss, _ = self.net_ql(zl)
                layout_quant_loss = layout_quant_loss * self.opt.lambda_quant
                if not self.opt.no_q_img:
                    loss += layout_quant_loss

        # merge intermediate features from image and layout
        # half of intermediate features are occupied by image features and half by layout features
        if self.opt.use_layout and self.opt.same_decoder_layout:
            for i in range(len(inter_enc)):
                half_size = inter_enc[i].size(2) // 2
                inter_enc[i] = torch.cat([inter_enc[i][:, :, :half_size], inter_encl[i][:, :, half_size:]], dim=2)
            z_q = torch.cat([z_q, zl_q], dim=2)

        # decode
        if self.opt.p2p_context:
            inters = [[feat[:, -1] for feat in inter_enc]]
            delta = 1
        else:
            inters = []
            delta = 0
        inters.append([feat[:, 0] for feat in inter_enc])
        fake_vid = [real_vid[:, 0]]
        if self.opt.use_layout and self.opt.same_decoder_layout:
            fake_layout = []
        curr = 1
        for i in range(1, self.opt.vid_len - delta):
            inter_tgts = [inters[-dt] for dt in self.opt.skip_context if dt <= curr]
            fake_img, fake_layout_img = self.net_g(z_q[:, i], inter_tgts)
            _, new_inter = self.net_e(fake_img)
            if self.opt.use_layout and self.opt.same_decoder_layout:
                fake_layout.append(fake_layout_img)
                _, new_interl = self.net_el(fake_layout_img)
                for i in range(len(new_inter)):
                    half_size = new_inter[i].size(1) // 2
                    new_inter[i] = torch.cat([new_inter[i][:, :half_size], new_interl[i][:, half_size:]], dim=1)
            if len(inters) >= self.opt.skip_memory:
                inters.pop(delta)
            else:
                curr += 1
            if len(inters) > 0:
                inters[-1] = [feat.detach() for feat in inters[-1]]
            inters.append(new_inter)
            fake_vid.append(fake_img)
        if self.opt.p2p_context:
            fake_vid.append(real_vid[:, -1])
        fake_vid = torch.stack(fake_vid, dim=1)
        if self.opt.use_layout and self.opt.same_decoder_layout:
            fake_layout = torch.stack(fake_layout, dim=1)

        # flatten videos for image-related losses
        real_img = real_vid[:, 1:].contiguous().view(-1, *real_vid.shape[2:])
        fake_img = fake_vid[:, 1:].contiguous().view(-1, *fake_vid.shape[2:])

        layout_ce_loss = None
        if self.opt.use_layout and self.opt.same_decoder_layout:
            real_layout = real_layout[:, 1:].contiguous().view(-1, *real_layout.shape[2:])
            fake_layout = fake_layout.contiguous().view(-1, *fake_layout.shape[2:])
            layout_ce_loss = self.layout_loss(fake_layout, real_layout.squeeze(1))
            loss += layout_ce_loss

        # direct recovery
        rec_loss = torch.mean(torch.abs(real_img - fake_img))
        if self.opt.use_direct_recovery_vid:
            loss += rec_loss

        # vgg features matching
        vgg_loss = None
        if self.opt.use_vgg_vid:
            vgg_loss = self.vgg_loss_img(fake_img, real_img) * self.opt.lambda_vgg
            loss += vgg_loss

        # adversarial
        gen_loss = None
        if self.opt.use_dv:
            score_fake = self.net_dv(fake_vid)["score"]
            gen_loss = self.gan_loss_vid.generator_loss_logits(score_fake).sum() * self.opt.lambda_gan
            loss += gen_loss

        # unconditional
        unc_gen_loss = None
        unc_per_loss = None
        fake_unc_img = torch.tensor([])
        fake_unc_vid = torch.tensor([])
        if self.opt.use_unc_gen:
            fake_unc_vid, _ = self.net_g(z_q, has_ctx=False)
            fake_unc_img = fake_unc_vid.view(-1, *fake_unc_vid.shape[2:])
            real_img = real_vid.view(-1, *real_vid.shape[2:])
            score_fake = self.net_di(fake_unc_img)["score"]
            unc_gen_loss = self.gan_loss_img.generator_loss_logits(score_fake).sum() * self.opt.lambda_gan
            loss += unc_gen_loss
            unc_per_loss = self.vgg_loss_img(fake_unc_img, real_img) * self.opt.lambda_vgg
            unc_per_loss += torch.mean(torch.abs(real_img - fake_unc_img))
            loss += unc_per_loss

        # adversarial
        gen_feat_loss = None
        if self.opt.use_df:
            score_real = self.net_df(z_q)["score"]
            gen_feat_loss = self.gan_loss_feat.generator_loss_logits_real(score_real).sum()
            loss += gen_feat_loss

        if self.logger:
            # log scalars every step
            self.logger.log_scalar("qvid_generator/quant_vid", quant_loss, global_iter)
            self.logger.log_scalar("qvid_generator/layout_quant_vid", layout_quant_loss, global_iter)
            self.logger.log_scalar("qvid_generator/gen_feat_real", gen_feat_loss, global_iter)
            self.logger.log_scalar("qvid_generator/gen_vid", gen_loss, global_iter)
            self.logger.log_scalar("qvid_generator/layout_vid", layout_ce_loss, global_iter)
            self.logger.log_scalar("qvid_generator/gen_img_unc", unc_gen_loss, global_iter)
            self.logger.log_scalar("qvid_generator/per_img_unc", unc_per_loss, global_iter)
            self.logger.log_scalar("qvid_generator/rec_vid", rec_loss, global_iter)
            self.logger.log_scalar("qvid_generator/vgg_vid", vgg_loss, global_iter)
            # log images every few steps
            if log:
                self.logger.log_vid("qvid_generator/fake_vid", fake_vid[:4].cpu(), global_iter, normalize=True, span=(-1, 1))
                self.logger.log_img("qvid_generator/fake_unc_img", fake_unc_img[:16].cpu(), 4, global_iter, normalize=True, span=(-1, 1))


        return loss, {"vid": fake_vid, "z": z_q, "unc_vid": fake_unc_vid}

    def compute_img_discriminator_loss(self, z, real_img, fake_img, log, global_iter):
        loss = torch.tensor(0., requires_grad=True).cuda()

        dis_loss = None
        if self.opt.elastic_corruption:
            n = self.opt.n_consecutive_img
            bs = real_img.size(0)
            no_corr_idx = [i for i in range(bs) if i % (n + 1) != 0]
            real_img = real_img[no_corr_idx]
        if self.opt.use_di:
            real_img_for_di = augment(real_img, self.ada_aug_p)[0] if self.opt.use_aug else real_img
            fake_img_for_di = augment(fake_img, self.ada_aug_p)[0] if self.opt.use_aug else fake_img
            fake_score = self.net_di(fake_img_for_di)["score"]
            real_score = self.net_di(real_img_for_di)["score"]
            dis_loss = self.gan_loss_img.discriminator_loss_logits(real_score, fake_score) * self.opt.lambda_gan
            loss += dis_loss

        feat_loss = None
        if self.opt.use_df:
            fake_score = self.net_df(z)["score"]
            feat_loss = self.gan_loss_feat.discriminator_loss_logits_fake(fake_score)
            loss += feat_loss

        r_t_stat = None
        if self.opt.use_aug and self.opt.aug_p == 0:
            self.ada_aug_p = self.ada_aug.tune(real_score.detach())
            r_t_stat = self.ada_aug.r_t_stat

        if self.logger:
            # log scalars every step
            self.logger.log_scalar("qvid_generator/dis_img", dis_loss, global_iter)
            self.logger.log_scalar("qvid_generator/dis_feat_fake", feat_loss, global_iter)
            self.logger.log_scalar("qvid_generator/rt_stat", r_t_stat, global_iter)
            # log images every few step
            if log:
                self.logger.log_img("qvid_generator/real_img", real_img[:16].float().cpu(), 4, global_iter, normalize=True, span=(-1, 1))

        return loss


    def compute_img_discriminator_regularization_loss(self, z, real_img, log, global_iter):
        if self.opt.elastic_corruption:
            n = self.opt.n_consecutive_img
            bs = real_img.size(0)
            no_corr_idx = [i for i in range(bs) if i % (n + 1) != 0]
            real_img = real_img[no_corr_idx]
        if self.opt.use_di:
            real_img.requires_grad = True
            real_img_for_di = augment(real_img, self.ada_aug_p)[0] if self.opt.use_aug else real_img
            real_score = self.net_di(real_img_for_di)["score"]
            grad_real, = autograd.grad(outputs=real_score.sum(), inputs=real_img, create_graph=True)
            grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
            # for details see https://github.com/rosinality/stylegan2-pytorch/issues/76
            loss = self.opt.lambda_r1 / 2 * grad_penalty * self.opt.d_reg_every + 0 * real_score[0]
        else:
            loss = torch.tensor(0., requires_grad=True).cuda()

        feat_loss = None
        if self.opt.use_df:
            z.requires_grad = True
            real_score = self.net_df(z)["score"]
            grad_real, = autograd.grad(outputs=real_score.sum(), inputs=z, create_graph=True)
            grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
            # for details see https://github.com/rosinality/stylegan2-pytorch/issues/76
            feat_loss = self.opt.lambda_r1 / 2 * grad_penalty * self.opt.d_reg_every + 0 * real_score[0]
            loss += feat_loss

        if self.logger:
            # log scalars every step
            self.logger.log_scalar("qvid_generator/dis_r1_loss", loss, global_iter)
            self.logger.log_scalar("qvid_generator/dis_r1_loss_img_feat", feat_loss, global_iter)

        return loss


    def compute_vid_discriminator_loss(self, z, real_vid, fake_vid, fake_unc_vid, log, global_iter):
        loss = torch.tensor(0., requires_grad=True).cuda()

        dis_loss = None
        if self.opt.use_dv:
            score_fake = self.net_dv(fake_vid)["score"]
            score_real = self.net_dv(real_vid)["score"]
            dis_loss = self.gan_loss_vid.discriminator_loss_logits(score_real, score_fake)
            loss += dis_loss

        # unconditional
        unc_dis_loss = None
        real_unc_img = torch.tensor([])
        if self.opt.use_unc_gen:
            real_unc_img = real_vid.view(-1, *real_vid.shape[2:])
            fake_unc_img = fake_unc_vid.view(-1, *fake_unc_vid.shape[2:])
            fake_score = self.net_di(fake_unc_img)["score"]
            real_score = self.net_di(real_unc_img)["score"]
            unc_dis_loss = self.gan_loss_img.discriminator_loss_logits(real_score, fake_score) * self.opt.lambda_gan
            loss += unc_dis_loss

        feat_loss = None
        if self.opt.use_df:
            real_score = self.net_df(z)["score"]
            feat_loss = self.gan_loss_feat.discriminator_loss_logits_real(real_score)
            loss += feat_loss

        if self.logger:
            # log scalars every step
            self.logger.log_scalar("qvid_generator/dis_vid", dis_loss, global_iter)
            self.logger.log_scalar("qvid_generator/dis_img_unc", unc_dis_loss, global_iter)
            self.logger.log_scalar("qvid_generator/dis_feat_real", feat_loss, global_iter)
            # log images every few step
            if log:
                self.logger.log_vid("qvid_generator/real_vid", real_vid[:4].cpu(), global_iter, normalize=True, span=(-1, 1))
                self.logger.log_img("qvid_generator/real_unc_img", real_unc_img[:16].float().cpu(), 4, global_iter, normalize=True, span=(-1, 1))

        return loss


    def compute_vid_discriminator_regularization_loss(self, z, real_vid, log, global_iter):
        if self.opt.use_dv:
            real_vid.requires_grad = True
            real_score = self.net_dv(real_vid)["score"]
            grad_real, = autograd.grad(outputs=real_score.sum(), inputs=real_vid, create_graph=True)
            grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
            # for details see https://github.com/rosinality/stylegan2-pytorch/issues/76
            loss = self.opt.lambda_r1 / 2 * grad_penalty * self.opt.d_reg_every + 0 * real_score[0]
        else:
            loss = torch.tensor(0., requires_grad=True).cuda()

        feat_loss = None
        if self.opt.use_df:
            z.requires_grad = True
            real_score = self.net_df(z)["score"]
            grad_real, = autograd.grad(outputs=real_score.sum(), inputs=z, create_graph=True)
            grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
            # for details see https://github.com/rosinality/stylegan2-pytorch/issues/76
            feat_loss = self.opt.lambda_r1 / 2 * grad_penalty * self.opt.d_reg_every + 0 * real_score[0]
            loss += feat_loss

        if self.logger:
            # log scalars every step
            self.logger.log_scalar("qvid_generator/vdis_r1_loss", loss, global_iter)
            self.logger.log_scalar("qvid_generator/vdis_r1_loss_vid_feat", feat_loss, global_iter)

        return loss


    @torch.no_grad()
    def encode(self, data, layout, dtype, log, suffix, global_iter):
        # encode rgb
        z, inter_enc = self.net_e(data)

        # encode layout
        has_layout = (layout is not None and 0 not in layout.size())
        if has_layout and self.opt.use_layout:
            if dtype == "vid":
                soft_layout = torch.zeros(*layout.shape[:2], self.opt.layout_size, *layout.shape[-2:]).cuda().scatter_(2, layout.unsqueeze(2), 1.0)
            else:
                soft_layout = torch.zeros(layout.size(0), self.opt.layout_size, *layout.shape[-2:]).cuda().scatter_(1, layout, 1.0)
            zl, inter_encl = self.net_el(soft_layout)
        else:
            zl, inter_encl = torch.tensor([]), torch.tensor([])

        # quantize rgb
        if self.opt.is_continuous:
            if self.opt.use_q_anyway:
                z, _, _ = self.net_q(z)
            # codes are embeddings
            code = z.transpose(-3, -1).transpose(-3, -2).contiguous()
            code = code.view(code.size(0), -1, code.size(-1))
        else:
            # codes are indices corresponding to nearest embeddings
            z, _, info = self.net_q(z)
            code = info[2].view(z.shape[0], -1)

        # quantize layout
        if has_layout and self.opt.use_layout:
            if self.opt.is_continuous:
                # codes are embeddings
                layout_code = zl.transpose(-3, -1).transpose(-3, -2).contiguous()
                layout_code = layout_code.view(layout_code.size(0), -1, layout_code.size(-1))
            else:
                zl, _, infol = self.net_ql(zl)
                layout_code = infol[2].view(zl.shape[0], -1)
        else:
            layout_code = torch.tensor([])

        if self.logger and log:
            if dtype == "vid":
                self.logger.log_vid("qvid_generator/real_vid" + suffix, data[:4].cpu(), global_iter, normalize=True, span=(-1, 1))
            else:
                self.logger.log_img("qvid_generator/real_img" + suffix, data[:16].cpu(), 4, global_iter, normalize=True, span=(-1, 1))

        return {"code": code, "state_code":layout_code, "inter": inter_enc, "interl": inter_encl, "z": z}


    @torch.no_grad()
    def decode(self, code, state_code, inter, interl, cond_inter, dtype, log, suffix, cond_frames, global_iter):
        fake_layout = None
        # convert codes to features
        shape = self.opt.z_shape[:2] if dtype == "img" else [-1] + self.opt.z_shape[:2]
        if self.opt.is_continuous:
            # codes are embeddings
            z = code.view(code.size(0), *shape, self.opt.z_size).transpose(-2, -1).transpose(-3, -2).contiguous()
        else:
            # codes are indices corresponding to nearest embeddings
            z = self.net_q.embed_code(code.view(-1, *self.opt.z_shape))
            z = z.view(code.size(0), *shape, self.opt.z_size).transpose(-2, -1).transpose(-3, -2).contiguous()

        # merge intermediate features from image and layout
        # half of intermediate features are occupied by image features and half by layout features
        if self.opt.use_layout and self.opt.same_decoder_layout:
            ctx = inter[0].size(1)
            ctxl = interl[0].size(1)
            zl = self.net_ql.embed_code(state_code.view(-1, *self.opt.z_shape))
            zl = zl.view(state_code.size(0), *shape, self.opt.z_size).transpose(-2, -1).transpose(-3, -2).contiguous()
            for i in range(len(inter)):
                half_size = inter[i].size(2) // 2
                inter[i] = torch.cat([inter[i][:, :ctx, :half_size], interl[i][:, :ctx, half_size:]], dim=2)
            print("z shape", z.shape, zl.shape)
            z = torch.cat([z, zl], dim=2)

        # decode
        if self.opt.use_inter and dtype == "vid" and self.opt.dec_model == "skipgan" and inter[0].size(1) < self.opt.vid_len:
            # decode frames from context window with their own inter
            ctx = inter[0].size(1)
            if ctx > 0:
                if self.opt.use_layout and self.opt.same_decoder_layout:
                    fake, fake_layout = self.net_g(z[:, :ctx].contiguous(), [inter])
                    fakes = [fake]
                    fake_layouts = [fake_layout]
                else:
                    fakes = [self.net_g(z[:, :ctx].contiguous(), [inter])[0]]
            else:
                fakes = []
                fake_layouts = []

            # propagate inter from previous frames to next frames
            pad_size = max(0, self.opt.skip_memory - ctx)
            pad_memory = [torch.zeros(code.size(0), pad_size, *feat.shape[2:]).cuda() for feat in inter]
            inter = [torch.cat([pad, feat[:, -self.opt.skip_memory:]], dim=1) for pad, feat in zip(pad_memory, inter)]
            curr = ctx
            if isinstance(cond_inter, list) and len(cond_inter) > 0:
                ctx += 1
            for _ in range(self.opt.vid_len - ctx):
                inters = [[feat[:, [-dt]] for feat in inter] for dt in self.opt.skip_context if dt <= curr]
                if isinstance(cond_inter, list) and len(cond_inter) > 0:
                    inters.append(cond_inter)
                fake_img, fake_layout, _, _, inter_dec = self.net_g(z[:, [curr]], inters, return_all=True, inter_pre_warping=False, has_ctx=curr > 0)
                if self.opt.skip_mode == "enc":
                    new_enc = self.encode(fake_img, (fake_layout.max(2)[1] if self.opt.use_layout and self.opt.same_decoder_layout else None), "vid", False, None, None)
                    new_inter = new_enc["inter"]
                    new_interl = new_enc["interl"]
                elif self.opt.skip_mode == "dec":
                    inter_dec.reverse()
                    new_inter = inter_dec
                else:
                    raise ValueError
                if self.opt.use_layout and self.opt.same_decoder_layout:
                    fake_layouts.append(fake_layout)
                    if curr < ctxl:
                        for i in range(len(inter)):
                            half_size = new_inter[i].size(2) // 2
                            new_inter[i] = torch.cat([new_inter[i][:, [0], :half_size], interl[i][:, [curr], half_size:]], dim=2)
                    else:
                        for i in range(len(inter)):
                            half_size = new_inter[i].size(2) // 2
                            new_inter[i] = torch.cat([new_inter[i][:, [0], :half_size], new_interl[i][:, [0], half_size:]], dim=2)

                for i in range(len(inter)):
                    if self.opt.keep_first and curr >= self.opt.skip_memory:
                        n = self.opt.n_first
                        inter[i][:, n:-1] = inter[i][:, n + 1:]  # .clone()
                    else:
                        inter[i][:, :-1] = inter[i][:, 1:] #.clone()
                    inter[i][:, [-1]] = new_inter[i]
                # inter = [torch.cat([feat[:, 1:], new_feat], dim=1) for feat, new_feat in zip(inter, new_inter)]
                fakes.append(fake_img)
                curr += 1
            fake = torch.cat(fakes, dim=1)
            if self.opt.use_layout and self.opt.same_decoder_layout:
                fake_layout = torch.cat(fake_layouts, dim=1)
        else:
            fake, _ = self.net_g(z, [inter])

        # display
        if self.logger and log:
            if dtype == "vid":
                self.logger.log_vid("qvid_generator/fake_vid" + suffix, fake[:4].cpu(), global_iter, normalize=True, span=(-1, 1), cond_frames=cond_frames)
            else:
                self.logger.log_img("qvid_generator/fake_img" + suffix, fake[:16].cpu(), 4, global_iter, normalize=True, span=(-1, 1))

        return {dtype: fake, "layout": fake_layout}

    @torch.no_grad()
    def vid_step_decode(self, code, inter, cond_inter):
        # convert codes to features
        shape = [1] + self.opt.z_shape[:2]
        if self.opt.is_continuous:
            # codes are embeddings
            z = code.view(code.size(0), *shape, self.opt.z_size).transpose(-2, -1).transpose(-3, -2).contiguous()
        else:
            # codes are indices corresponding to nearest embeddings
            z = self.net_q.embed_code(code.view(-1, *self.opt.z_shape))
            z = z.view(code.size(0), *shape, self.opt.z_size).transpose(-2, -1).transpose(-3, -2).contiguous()

        # decode
        assert self.opt.use_inter and self.opt.dec_model == "skipgan"
        ctx = inter[0].size(1)
        inters = [[feat[:, [-dt]] for feat in inter] for dt in self.opt.skip_context if dt <= ctx]
        if isinstance(cond_inter, list) and len(cond_inter) > 0:
            inters.append(cond_inter)
        fake, _, _, _, _ = self.net_g(z, inters, return_all=True, inter_pre_warping=False)
        new_data = self.encode(fake, None, None, False, None, None)
        new_inter = new_data["inter"]
        code = new_data["code"]
        if ctx < self.opt.skip_memory:
            inter = [torch.cat([feat, new_feat], dim=1) for feat, new_feat in zip(inter, new_inter)]
        else:
            for i in range(len(inter)):
                inter[i][:, :-1] = inter[i][:, 1:]
                inter[i][:, [-1]] = new_inter[i]

        return {"vid": fake, "inter": inter, "code": code}

    def accumulate(self, decay=0.999):
        def acc(model1, model2, decay):
            par1 = dict(model1.named_parameters())
            par2 = dict(model2.named_parameters())
            for k in par1.keys():
                par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
        if self.opt.use_ema:
            acc(self.net_e_ema, self.net_e, decay)
            acc(self.net_el_ema, self.net_el, decay) if self.net_el is not None else None
            if not self.opt.is_continuous or self.opt.use_q_anyway:
                acc(self.net_q_ema, self.net_q, decay)
                acc(self.net_ql_ema, self.net_ql, decay) if self.net_ql is not None else None
            acc(self.net_g_ema, self.net_g, decay)
            acc(self.net_gl_ema, self.net_gl, decay) if self.net_gl is not None else None





