import torch
import torch.nn.functional as F

from ..models.mingpt import GPT, CGPT, NoiseInjection
from tools.utils import to_cuda
from models import load_network, save_network, print_network
from tqdm import tqdm
from ..modules.vmf import nll_vMF

class Transformer(torch.nn.Module):
    def __init__(self, opt, is_train=True, is_main=True, logger=None):
        super().__init__()
        self.opt = opt
        self.is_main = is_main

        self.net_t = self.initialize_networks(is_train)

        if is_train:
            self.opt_t = self.create_optimizers(self.opt)

        self.logger = logger if self.is_main else None

        height, width = self.opt.z_shape
        self.size = height * width
        self.state_size = self.opt.state_size
        self.tot_size = self.size + self.state_size


    def forward(self, data, prefix='', mode='', total_len=None, log=False, global_iter=None, show_progress=False):
        code, state_code, cond_code, delta_length_cond, vid_lbl = self.preprocess_input(data)

        if mode == 'transformer':
            t_loss = self.compute_transformer_loss(code, state_code, cond_code, delta_length_cond, vid_lbl, prefix, log, global_iter)
            return t_loss

        if mode == 'eval_transformer':
            with torch.no_grad():
                t_loss = self.compute_transformer_loss(code, log, global_iter, is_eval=True)
            return t_loss

        if mode == 'inference':
            return self.generate_fake(code, state_code, cond_code, delta_length_cond, vid_lbl, total_len, show_progress)

        else:
            raise ValueError(f"mode '{mode}' is invalid")


    def preprocess_input(self, data):
        data["code"] = to_cuda(data, "code", flatten_empty=False)
        data["state_code"] = to_cuda(data, "state_code", flatten_empty=False)
        data["cond_code"] = to_cuda(data, "cond_code")
        data["vid_lbl"] = to_cuda(data, "vid_lbl")
        data["delta_length_cond"] = to_cuda(data, "delta_length_cond")
        return data["code"], data["state_code"], data["cond_code"], data["delta_length_cond"], data["vid_lbl"]


    def initialize_networks(self, is_train):
        if self.opt.is_continuous:
            net_t = CGPT(n_proposals=self.opt.n_proposals, block_size=self.opt.z_len, n_layer=self.opt.n_layer,
                         n_head=self.opt.n_head, n_embd=self.opt.n_embd, n_in=self.opt.n_in,
                         resid_noise=self.opt.resid_noise).cuda()
        else:
            num_lbl = len(self.opt.categories) if self.opt.categories is not None else None
            net_t = GPT(vocab_size=self.opt.z_num, block_size=self.opt.z_len, n_layer=self.opt.n_layer,
                        n_head=self.opt.n_head, n_embd=self.opt.n_embd, emb_mode=self.opt.emb_mode,
                        shape=self.opt.z_shape, state_vocab_size=self.opt.state_num, num_blocks=self.opt.num_blocks,
                        state_size=self.opt.state_size, use_start_token=self.opt.use_start_token, use_lbl=self.opt.cat,
                        num_lbl=num_lbl, state_front=self.opt.state_front).cuda()

        if self.is_main:
            net_t = load_network(net_t, "transformer_t", self.opt, head_to_n=self.opt.head_to_n)

        return net_t


    def save_model(self, global_iter, latest=False, best=False):
        save_network(self.net_t, "transformer_t", global_iter, self.opt, latest, best)


    # Following minGPT:
    # This long function is unfortunately doing something very simple and is being very defensive:
    # We are separating out all parameters of the model into two buckets: those that will experience
    # weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    # We are then returning the PyTorch optimizer object.
    def create_optimizers(self, opt):
        param_dict = {pn: p for pn, p in self.net_t.named_parameters()}
        if opt.finetune_head and opt.finetune_f is None:
            optim_groups = [{"params": [param_dict["head.weight"]], "weight_decay": 0.01, "lr": opt.lr}]
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear,)
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, NoiseInjection)
            for mn, m in self.net_t.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)

            # special case the position embedding parameter in the root GPT module as not decayed
            no_decay.add('start_tok_emb') if 'start_tok_emb' in param_dict.keys() else None
            no_decay.add('pos_emb') if 'pos_emb' in param_dict.keys() else None
            no_decay.add('h_emb') if 'h_emb' in param_dict.keys() else None
            no_decay.add('w_emb') if 'w_emb' in param_dict.keys() else None
            no_decay.add('s_emb') if 's_emb' in param_dict.keys() else None
            no_decay.add('t_emb') if 't_emb' in param_dict.keys() else None
            no_decay.add('state_pos_emb') if 'state_pos_emb' in param_dict.keys() else None
            no_decay.add('state_s_emb') if 'state_s_emb' in param_dict.keys() else None

            # validate that we considered every parameter
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

            # create the pytorch optimizer object
            if opt.finetune_head:
                optim_groups = [{"params": [param_dict[pn] for pn in sorted(list(decay)) if pn != "head.weight"], "weight_decay": 0.01, "lr": opt.lr * opt.finetune_f},
                                {"params": [param_dict["head.weight"]], "weight_decay": 0.01, "lr": opt.lr},
                                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0, "lr": opt.lr * opt.finetune_f}]
            else:
                optim_groups = [{"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01, "lr": opt.lr},
                                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0, "lr": opt.lr}]


        if opt.optimizer == "adamw":
            opt_t = torch.optim.AdamW(optim_groups, betas=(opt.beta1, opt.beta2))
        else:
            raise NotImplementedError
        return opt_t


    def compute_transformer_loss(self, code, state_code, cond_code, delta_length_cond, vid_lbl, prefix, log, global_iter, is_eval=False):
        code = code[:, :self.opt.z_len] # limit input to transformer capacity

        state_nll_loss = None

        if self.opt.is_continuous:
            if self.opt.p2p:
                pred = self.net_t(code[:, :-1], cond_code, delta_length_cond, lbl_idx=vid_lbl)
            else:
                pred = self.net_t(code[:, :-1], lbl_idx=vid_lbl)
            tgt = code[:, 1:]
            vmf_loss = None
            other_vmf_loss = None
            cosine_loss = None
            other_cosine_loss = None
            # nll_loss = None

            nll_loss = F.mse_loss(pred, tgt)
            t_loss = nll_loss

            # if self.opt.n_proposals > 1:
            #     t_loss = torch.tensor(0., requires_grad=True).cuda()
            #     logits, proposals = pred
            #     nm_proposals = proposals / torch.norm(proposals, p=2, dim=3, keepdim=True) if self.opt.normalize_pred else proposals
            #     nm_tgt = tgt / torch.norm(tgt, p=2, dim=2, keepdim=True) if self.opt.normalize_tgt else tgt
            #     cosine_dist = - (nm_proposals * nm_tgt.unsqueeze(2)).sum(dim=3)
            #     closest_proposals = cosine_dist.argmin(dim=2, keepdim=True)
            #     nll_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), closest_proposals.view(-1))
            #     t_loss += nll_loss
            #     if self.opt.knn is not None:
            #         k_closest = max(1, int(self.opt.knn * (1 - global_iter / self.opt.knn_decay_iter)))
            #         closest_proposals = (-cosine_dist).topk(dim=2, k=k_closest)[1]
            #     else:
            #         k_closest = 1
            #     closest_onehot = torch.zeros(*closest_proposals.shape[:2], self.opt.n_proposals).cuda().scatter_(2, closest_proposals, 1)
            #     if self.opt.continuous_loss == "cosine":
            #         pred = nm_proposals[closest_onehot.bool()].view(*nm_proposals.shape[:2], k_closest, -1)
            #         cosine_loss = - (pred * tgt.unsqueeze(2)).sum(dim=3).mean()
            #         if self.opt.knn is not None:
            #             t_loss += cosine_loss
            #         else:
            #             other_preds = nm_proposals[~closest_onehot.bool()].view(*nm_proposals.shape[:2], self.opt.n_proposals - k_closest, -1)
            #             other_cosine_loss = - (other_preds * tgt.unsqueeze(2)).sum(dim=3).mean()
            #             t_loss += (1 - self.opt.epsilon_other) * cosine_loss + self.opt.epsilon_other * other_cosine_loss
            #     elif self.opt.continuous_loss == "vmf":
            #         pred = proposals[closest_onehot.bool()].view(*nm_proposals.shape[:2], k_closest, -1)
            #         vmf_loss = nll_vMF(pred, tgt.unsqueeze(2))
            #         if self.opt.knn is not None:
            #             t_loss += vmf_loss
            #         else:
            #             other_preds = proposals[~closest_onehot.bool()].view(*nm_proposals.shape[:2], self.opt.n_proposals - k_closest, -1)
            #             other_vmf_loss = nll_vMF(other_preds, tgt.unsqueeze(2))
            #             t_loss += (1 - self.opt.epsilon_other) * vmf_loss + self.opt.epsilon_other * other_vmf_loss
            #
            # else:
            #     if self.opt.continuous_loss == "cosine":
            #         if self.opt.normalize_pred:
            #             pred = pred / torch.norm(pred, p=2, dim=2, keepdim=True)
            #         if self.opt.normalize_tgt:
            #             tgt = tgt / torch.norm(tgt, p=2, dim=2, keepdim=True)
            #         cosine_loss = - (pred * tgt).sum(dim=2).mean()
            #         t_loss = cosine_loss
            #     elif self.opt.continuous_loss == "vmf":
            #         vmf_loss = nll_vMF(pred, tgt)
            #         t_loss = vmf_loss

            nrec_loss = None
            nrec_momentum_loss = None

        else:
            logits = self.net_t(code[:, :-1], cond_idx=cond_code, state_idx=state_code, delta_length_cond=delta_length_cond, lbl_idx=vid_lbl)

            if 0 not in state_code.size():
                if self.opt.state_front:
                    state_i = [i for i in range(logits.size(1)) if (i + 1) < self.state_size * self.opt.num_blocks]
                    frame_i = [i for i in range(logits.size(1)) if (i + 1) >= self.state_size * self.opt.num_blocks]
                else:
                    state_i = [i for i in range(logits.size(1)) if (i + 1) % self.tot_size < self.state_size]
                    frame_i = [i for i in range(logits.size(1)) if (i + 1) % self.tot_size >= self.state_size]
                state_logits = logits[:, state_i, :self.opt.state_num]
                logits = logits[:, frame_i]
                target = code
            else:
                if self.opt.use_start_token or self.opt.cat:
                    target = code
                else:
                    target = code[:, 1:]
            nll_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            nrec_loss = None
            other_vmf_loss = None
            cosine_loss = None
            other_cosine_loss = None
            nrec_momentum_loss = None
            vmf_loss = None
            t_loss = nll_loss

            if 0 not in state_code.size():
                state_nll_loss = F.cross_entropy(state_logits.reshape(-1, state_logits.size(-1)), state_code[:, 1:].reshape(-1))
                t_loss += state_nll_loss

        if self.logger and not is_eval:
            # log scalars every step
            self.logger.log_scalar(f"transformer/{prefix}nll", nll_loss, global_iter)
            self.logger.log_scalar(f"transformer/{prefix}state_nll", state_nll_loss, global_iter)
            self.logger.log_scalar(f"transformer/{prefix}cosine", cosine_loss, global_iter)
            self.logger.log_scalar(f"transformer/{prefix}other_cosine", other_cosine_loss, global_iter)
            self.logger.log_scalar(f"transformer/{prefix}vmf", vmf_loss, global_iter)
            self.logger.log_scalar(f"transformer/{prefix}other_vmf", other_vmf_loss, global_iter)
            self.logger.log_scalar(f"transformer/{prefix}nrec", nrec_loss, global_iter)
            self.logger.log_scalar(f"transformer/{prefix}nrec_momentum", nrec_momentum_loss, global_iter)

        return t_loss


    def top_k_logits(self, logits, k):
        v, _ = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out


    @torch.no_grad()
    def generate_fake(self, code, state_code, cond_code, delta_length_cond, vid_lbl, total_len, show_progress):
        ''' If 'total_len' is 'None' generate tokens with transformer until the capacity 'z_len' of the transformer has
        been reached. Otherwise, fill the code until 'total_len' is reached with a 'z_chunk' stride.
        '''

        if total_len is None:
            code, state_code = self.fill_code(code, state_code, cond_code, delta_length_cond, vid_lbl, show_progress=show_progress)
            return {"code": code, "state_code": state_code}

        if total_len <= self.opt.z_len:
            add_len = total_len - code.size(1)
            add_len -= cond_code.size(1) if 0 not in cond_code.size() else 0
            add_len -= min(state_code.size(1), self.opt.state_size * self.opt.num_blocks) if 0 not in state_code.size() else 0
            code, state_code = self.fill_code(code, state_code, cond_code, delta_length_cond, vid_lbl, add_len=add_len, show_progress=show_progress)
            return {"code": code, "state_code": state_code}

        if show_progress:
            pbar = tqdm(total=int(total_len), desc="Processing codes")

        # 1. fill until transformer capacity 'z_len' is reached
        code, state_code = self.fill_code(code, state_code, cond_code, delta_length_cond, vid_lbl, show_progress=show_progress)

        # 2. predict 'z_chunk' by 'z_chunk'
        curr_len = self.opt.z_len
        if show_progress:
            pbar.update(curr_len)

        i = 1
        while curr_len < total_len:
            add_len = total_len - curr_len if total_len - curr_len < self.opt.z_chunk else None

            if 0 not in cond_code.size():
                delta_length_cond -= 1

            # free some capacity for one chunk
            tmp_state_code = state_code[:, i * self.state_size:] if 0 not in state_code.size() else state_code
            tmp_code = code[:, i * self.size:]

            # predict one chunk
            pred_code, pred_state_code = self.fill_code(tmp_code, tmp_state_code, cond_code, delta_length_cond, vid_lbl, add_len=add_len, show_progress=show_progress)

            # update code
            delta_code = pred_code.size(1) - tmp_code.size(1)
            code = torch.cat([code, pred_code[:, -delta_code:]], dim=1)
            if 0 not in state_code.size():
                delta_state_code = pred_state_code.size(1) - tmp_state_code.size(1)
                if delta_state_code > 0:
                    state_code = torch.cat([state_code, pred_state_code[:, -delta_state_code:]], dim=1)
                # else:
                #     curr_len += self.state_size

            # keep track of progress
            curr_len += add_len if add_len is not None else self.opt.z_chunk
            if show_progress:
                # if add_len is not None:
                #     print("add_len", add_len)
                # else:
                #     print("z_chunk", self.opt.z_chunk)
                pbar.update(add_len if add_len is not None else self.opt.z_chunk)
            i += 1

        if show_progress:
            pbar.close()

        return {"code": code, "state_code": state_code}


    def fill_code(self, code, state_code, cond_code, delta_length_cond, vid_lbl, add_len=None, show_progress=False):
        bs = code.size(0)
        log_p = None

        # compute add_len
        if add_len is None:
            add_len = self.opt.z_len - code.size(1)
            add_len -= cond_code.size(1) if 0 not in cond_code.size() else 0
            add_len -= min(state_code.size(1), self.opt.state_size * self.opt.num_blocks) if 0 not in state_code.size() else 0

        # iterate
        pbar = tqdm(range(add_len), desc="Filling codes", leave=False) if show_progress else range(add_len)
        for _ in pbar:
            if self.opt.is_continuous:
                pred = self.net_t(code, single=True)
                if self.opt.normalize_pred:
                    pred = pred / torch.norm(pred, p=2, dim=2, keepdim=True)
                code = torch.cat((code, pred), dim=1)
            else:
                logits = self.net_t(code, cond_idx=cond_code, state_idx=state_code, delta_length_cond=delta_length_cond, lbl_idx=vid_lbl)
                # determine if prediction needs to be affected to code or state_code
                is_state = 0 not in state_code.size() and logits.size(1) % self.tot_size < self.state_size
                if is_state:
                    logits = logits[:, :, :self.opt.state_num]
                    icode = self.get_icode(logits, self.opt.temperature_state, self.opt.top_k_state, self.opt.sample_state)[0]
                    state_code = torch.cat((state_code, icode), dim=1)
                else:
                    if self.opt.beam_size is not None:
                        if code.size(0) == bs:
                            # expand
                            code = code.unsqueeze(1).repeat(1, self.opt.beam_size, 1).view(bs * self.opt.beam_size, -1)
                            icode, ilog_p = self.get_icode(logits, self.opt.temperature, self.opt.top_k, self.opt.sample, n=self.opt.beam_size)
                            log_p = ilog_p
                            icode = icode.view(-1, 1)
                        else:
                            if not self.opt.no_sample:
                                icode, ilog_p = self.get_icode(logits, self.opt.temperature, self.opt.top_k, self.opt.sample, n=1)
                                log_p += ilog_p.view(bs, self.opt.beam_size)
                                icode = icode.view(-1, 1)
                            else:
                                # expand
                                icode, ilog_p = self.get_icode(logits, self.opt.temperature, self.opt.top_k, self.opt.sample, n=self.opt.beam_size)
                                log_p = log_p.unsqueeze(1).repeat(1, self.opt.beam_size, 1)
                                log_p += ilog_p.view(bs, self.opt.beam_size, self.opt.beam_size)
                                icode = icode.view(bs, self.opt.beam_size * self.opt.beam_size)
                                log_p = log_p.view(bs, self.opt.beam_size * self.opt.beam_size)
                                # prune
                                log_p, keep = torch.topk(log_p, dim=1, k=self.opt.beam_size)
                                icode = torch.gather(icode, dim=1, index=keep).view(-1, 1)
                                code = code.unsqueeze(1).repeat(1, self.opt.beam_size, 1).view(bs, self.opt.beam_size * self.opt.beam_size, -1)
                                keep = keep.unsqueeze(-1).repeat(1, 1, code.size(-1))
                                code = torch.gather(code, dim=1, index=keep).view(-1, code.size(-1))
                    else:
                        icode = self.get_icode(logits, self.opt.temperature, self.opt.top_k, self.opt.sample)[0]
                    code = torch.cat((code, icode), dim=1)
        if self.opt.beam_size is not None:
            # keep best hypothesis
            _, best = torch.topk(log_p, dim=1, k=1)
            code = code.view(bs, self.opt.beam_size, -1)
            best = best.unsqueeze(-1).repeat(1, 1, code.size(-1))
            code = torch.gather(code, dim=1, index=best).view(bs, code.size(-1))
        return code, state_code


    def get_icode(self, logits, temperature, top_k, sample, n=1):
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            icode = torch.multinomial(probs, num_samples=n)
        else:
            _, icode = torch.topk(probs, k=n, dim=-1)
        ilog_p = torch.log(torch.gather(probs, 1, icode))
        return icode, ilog_p