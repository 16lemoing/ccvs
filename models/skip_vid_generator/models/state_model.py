import torch
import torch.nn.functional as F

from ..models.skip_autoencoder import StateEstimator
from ..modules.quantize import VectorQuantizer
from tools.utils import to_cuda, DummyOpt
from models import load_network, save_network, print_network


class StateModel(torch.nn.Module):
    def __init__(self, opt, is_train=True, is_main=True, logger=None):
        super().__init__()
        self.opt = opt
        self.is_main = is_main

        self.initialize_networks(is_train)

        if is_train:
            self.opt_s = self.create_optimizers(self.opt)

        self.logger = logger if self.is_main else None

    def forward(self, data, mode='', log=False, global_iter=None):
        z, state, state_code = self.preprocess_input(data)

        if mode == 'state_estimator':
            s_loss = self.compute_state_estimator_loss(z, state, log, global_iter)
            return s_loss

        if mode == 'eval_state_estimator':
            s_loss = self.compute_eval_state_estimator_loss(z, state, log, global_iter)
            return s_loss

        if mode == 'img_encoder':
            return self.encode(state, z)

        if mode == 'vid_encoder':
            return self.encode(state, z)

        if mode == 'img_decoder':
            return self.decode(state_code, "img")

        if mode == 'vid_decoder':
            return self.decode(state_code, "vid")

        else:
            raise ValueError(f"mode '{mode}' is invalid")

    def preprocess_input(self, data):
        data["z"] = to_cuda(data, "z")
        data["state"] = to_cuda(data, "state")
        data["state_code"] = to_cuda(data, "state_code")
        return data["z"], data["state"], data["state_code"]

    def initialize_networks(self, is_train):
        self.net_s = StateEstimator(self.opt).cuda() if not self.opt.quantize_only else None
        self.net_q = VectorQuantizer(self.opt.state_num, 1, beta=0.25).cuda()
        if self.is_main:
            self.net_s = load_network(self.net_s, "state_s", self.opt) if self.net_s is not None else None
            self.net_q = load_network(self.net_q, "state_q", self.opt)
            if True:
                print_network(self.net_q)
                print_network(self.net_s)

    def save_model(self, global_iter, latest=False, best=False, state_only=False):
        save_network(self.net_q, "state_q", global_iter, self.opt, latest, best)
        save_network(self.net_s, "state_s", global_iter, self.opt, latest, best)

    def create_optimizers(self, opt):
        if opt.optimizer == "adam":
            s_params = list(self.net_q.parameters())
            s_params += list(self.net_s.parameters()) if self.net_s is not None else []
            opt_s = torch.optim.Adam(s_params, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
        else:
            raise NotImplementedError
        return opt_s

    def compute_state_estimator_loss(self, z, state, log, global_iter):
        loss = torch.tensor(0., requires_grad=True).cuda()

        if self.net_s is not None:
            pred_state = self.net_s(z)
            rec_loss = F.mse_loss(pred_state, state)
            loss += rec_loss
        else:
            rec_loss = None
            pred_state = state

        _, quant_loss, _ = self.net_q(pred_state)
        loss += quant_loss

        if self.logger:
            # log scalars every step
            self.logger.log_scalar("state_estimator/rec", rec_loss, global_iter)
            self.logger.log_scalar("state_estimator/quant", quant_loss, global_iter)

        return loss

    @torch.no_grad()
    def compute_eval_state_estimator_loss(self, z, state, log, global_iter):
        if self.net_s is not None:
            pred_state = self.net_s(z)
        else:
            pred_state = state
        pred_state_q, quant_loss, _ = self.net_q(pred_state)
        loss = F.mse_loss(pred_state_q, state)
        return loss

    @torch.no_grad()
    def encode(self, state, z):
        if 0 in state.size():
            # estimate state
            state = self.net_s(z)
        # quantize
        state_q, _, info = self.net_q(state)
        code = info[2].view(state.shape[0], -1)
        return {"state_code": code}

    @torch.no_grad()
    def decode(self, state_code, dtype):
        shape = [self.opt.state_size] if dtype == "img" else [-1, self.opt.state_size]
        state = self.net_q.embed_code(state_code)
        state = state.view(state_code.size(0), *shape)
        return {"state": state}