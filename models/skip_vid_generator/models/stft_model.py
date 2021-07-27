import torch
import torch.nn.functional as F

from ..models.skip_autoencoder import StftEncoder, StftDecoder
from ..modules.quantize import VectorQuantizer
from ..modules.perceptual import VGGLoss
from tools.utils import to_cuda, DummyOpt
from models import load_network, save_network, print_network


class StftModel(torch.nn.Module):
    def __init__(self, opt, is_train=True, is_main=True, logger=None):
        super().__init__()
        self.opt = opt
        self.is_main = is_main

        self.initialize_networks(is_train)

        if is_train:
            self.opt_s = self.create_optimizers(self.opt)
            self.vgg_loss = VGGLoss(self.opt.gpu_ids)

        self.logger = logger if self.is_main else None

    def forward(self, data, mode='', log=False, global_iter=None):
        stft, stft_code = self.preprocess_input(data)

        if mode == 'stft_reconstruction':
            s_loss = self.compute_stft_reconstruction_loss(stft, log, global_iter)
            return s_loss

        if mode == 'eval_stft_reconstruction':
            s_loss = self.compute_eval_stft_reconstruction_loss(stft, log, global_iter)
            return s_loss

        if mode == 'img_encoder':
            return self.encode(stft)

        if mode == 'vid_encoder':
            return self.encode(stft)

        if mode == 'img_decoder':
            return self.decode(stft_code, "img")

        if mode == 'vid_decoder':
            return self.decode(stft_code, "vid")

        else:
            raise ValueError(f"mode '{mode}' is invalid")

    def preprocess_input(self, data):
        data["stft"] = to_cuda(data, "stft")
        data["state_code"] = to_cuda(data, "state_code")
        return data["stft"], data["state_code"]

    def initialize_networks(self, is_train):
        self.net_e = StftEncoder(self.opt).cuda()
        self.net_d = StftDecoder(self.opt).cuda()
        self.net_q = VectorQuantizer(self.opt.stft_num, self.opt.stft_size, beta=0.25).cuda()
        if self.is_main:
            self.net_e = load_network(self.net_e, "stft_e", self.opt)
            self.net_d = load_network(self.net_d, "stft_d", self.opt)
            self.net_q = load_network(self.net_q, "stft_q", self.opt)
            if True:
                print_network(self.net_q)
                print_network(self.net_e)
                print_network(self.net_d)

    def save_model(self, global_iter, latest=False, best=False):
        save_network(self.net_q, "stft_q", global_iter, self.opt, latest, best)
        save_network(self.net_e, "stft_e", global_iter, self.opt, latest, best)
        save_network(self.net_d, "stft_d", global_iter, self.opt, latest, best)

    def create_optimizers(self, opt):
        if opt.optimizer == "adam":
            s_params = list(self.net_q.parameters())
            s_params += list(self.net_e.parameters())
            s_params += list(self.net_d.parameters())
            opt_s = torch.optim.Adam(s_params, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
        else:
            raise NotImplementedError
        return opt_s

    def compute_stft_reconstruction_loss(self, stft, log, global_iter):
        loss = torch.tensor(0., requires_grad=True).cuda()

        z = self.net_e(stft)

        z_q, quant_loss, _ = self.net_q(z)
        quant_loss = quant_loss
        loss += quant_loss

        stft_pred = self.net_d(z_q)

        rec_loss = F.mse_loss(stft, stft_pred)
        vgg_loss = self.vgg_loss(stft.view(-1, *stft.shape[2:]).repeat(1, 3, 1, 1), stft_pred.view(-1, *stft_pred.shape[2:]).repeat(1, 3, 1, 1))
        loss += rec_loss
        loss += vgg_loss

        if self.logger:
            # log scalars every step
            self.logger.log_scalar("state_estimator/rec", rec_loss, global_iter)
            self.logger.log_scalar("state_estimator/vgg", vgg_loss, global_iter)
            self.logger.log_scalar("state_estimator/quant", quant_loss, global_iter)
            # log images every few steps
            if log:
                self.logger.log_img("qvid_generator/stft", stft.cpu()[:16, 0], 4, global_iter, normalize=True, span=(-1, 1))
                self.logger.log_img("qvid_generator/stft_pred", stft_pred.cpu()[:16, 0], 4, global_iter, normalize=True, span=(-1, 1))

        return loss

    @torch.no_grad()
    def compute_eval_stft_reconstruction_loss(self, stft, log, global_iter):
        z = self.net_e(stft)
        z_q, _, _ = self.net_q(z)
        stft_pred = self.net_d(z_q)
        loss = F.mse_loss(stft, stft_pred)
        return loss

    @torch.no_grad()
    def encode(self, stft):
        z = self.net_e(stft)
        _, _, info = self.net_q(z)
        code = info[2].view(stft.shape[0], -1)
        return {"state_code": code}

    @torch.no_grad()
    def decode(self, state_code, dtype):
        shape = self.opt.stft_shape if dtype == "img" else [-1] + self.opt.stft_shape
        z = self.net_q.embed_code(state_code.view(-1, *self.opt.stft_shape))
        z = z.view(state_code.size(0), *shape, self.opt.stft_size).transpose(-2, -1).transpose(-3, -2).contiguous()
        stft = self.net_d(z)
        return {"stft": stft}