# Adapted from: https://github.com/karpathy/minGPT/

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class GPT2Config(GPTConfig):
    """ GPT-2 like network roughly 1.5B params """
    # TODO


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask_size = config.block_size
        # if config.state_size > 0:
        #     height, width = config.shape
        #     mask_size += config.block_size // (height * width) * config.state_size
        mask = torch.tril(torch.ones(mask_size, mask_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, mask_size, mask_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class NoiseInjection(nn.Module):
    def __init__(self, use_noise):
        super().__init__()
        self.use_noise = use_noise
        if self.use_noise:
            self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if self.use_noise:
            b, t, c = x.shape
            noise = x.new_empty(b, t, 1).normal_()
            return x + self.weight * noise
        else:
            return x

class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            NoiseInjection(config.resid_noise),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, vocab_size, block_size, num_blocks, n_layer=12, n_head=8, n_embd=256, embd_pdrop=0., resid_pdrop=0.,
                 attn_pdrop=0., n_unmasked=0, resid_noise=False, emb_mode=None, shape=None, state_vocab_size=0, state_size=0,
                 use_start_token=False, num_lbl=0, use_lbl=False, state_front=False):
        super().__init__()
        config = GPTConfig(block_size=block_size, vocab_size=vocab_size, embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop,
                           attn_pdrop=attn_pdrop, n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_unmasked=n_unmasked,
                           resid_noise=resid_noise, shape=shape, emb_mode=emb_mode, state_vocab_size=state_vocab_size,
                           state_size=state_size, use_start_token=use_start_token, num_blocks=num_blocks, num_lbl=num_lbl,
                           use_lbl=use_lbl, state_front=state_front)

        # token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        if config.state_vocab_size > 0:
            self.state_tok_emb = nn.Embedding(config.state_vocab_size, config.n_embd)
        if config.use_start_token:
            self.start_tok_emb = nn.Parameter(torch.randn(1, config.n_embd))

        # label embedding
        if config.use_lbl:
            self.lbl_emb = nn.Embedding(config.num_lbl, config.n_embd)

        # positional embedding
        height, width = config.shape
        if config.emb_mode is not None:
            # assert config.block_size % (height * width) == 0
            if config.emb_mode == "spatio-temporal":
                self.h_emb = nn.Parameter(torch.zeros(1, height, config.n_embd))
                self.w_emb = nn.Parameter(torch.zeros(1, width, config.n_embd))
                self.t_emb = nn.Parameter(torch.zeros(1, config.num_blocks, config.n_embd))
            elif config.emb_mode == "temporal":
                self.s_emb = nn.Parameter(torch.zeros(1, height * width, config.n_embd))
                self.t_emb = nn.Parameter(torch.zeros(1, config.num_blocks, config.n_embd))
            else:
                raise ValueError
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.num_blocks * height * width, config.n_embd))
        if config.state_size > 0:
            if config.emb_mode is not None:
                self.state_s_emb = nn.Parameter(torch.zeros(1, config.state_size, config.n_embd))
            else:
                self.state_pos_emb = nn.Parameter(torch.zeros(1, config.num_blocks * config.state_size, config.n_embd))

        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, max(config.vocab_size, config.state_vocab_size), bias=False)
        self.block_size = config.block_size + (1 if config.use_start_token else 0) + (1 if config.use_lbl else 0)
        self.apply(self._init_weights)
        self.config = config

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_pos_emb(self, t, delta_length=None):
        if t == 0:
            return 0
        height, width = self.config.shape
        size = height * width
        if delta_length is None or 0 in delta_length.size():
            delta_length = torch.zeros(1).long().cuda()
        n = delta_length.size(0)
        if self.config.emb_mode is not None:
            length = t // size + (1 if t % size != 0 else 0)
            if self.config.emb_mode == "spatio-temporal":
                pos_emb = self.h_emb.view(1, 1, height, 1, -1).repeat(1, length, 1, width, 1)
                pos_emb = pos_emb + self.w_emb.view(1, 1, 1, width, -1)
                t_emb = torch.zeros_like(self.t_emb[:, :length]).view(1, length, 1, 1, -1).repeat(n, 1, 1, 1, 1)
                for i in range(n):
                    t_emb[i] = self.t_emb[:, delta_length[i]:delta_length[i] + length].view(length, 1, 1, -1)
                pos_emb = pos_emb + t_emb
            elif self.config.emb_mode == "temporal":
                pos_emb = self.s_emb.view(1, 1, size, -1).repeat(1, length, 1, 1)
                t_emb = torch.zeros_like(self.t_emb[:, :length]).view(1, length, 1, -1).repeat(n, 1, 1, 1)
                for i in range(n):
                    t_emb[i] = self.t_emb[:, delta_length[i]:delta_length[i] + length].view(length, 1, -1)
                pos_emb = pos_emb + t_emb
            else:
                raise ValueError
            pos_emb = pos_emb.view(n, length * size, -1)
            pos_emb = pos_emb[:, :t, :]
        else:
            pos_emb = torch.zeros_like(self.pos_emb[:, :t]).view(1, t, -1).repeat(n, 1, 1)
            for i in range(n):
                pos_emb[i] = self.pos_emb[:, delta_length[i] * size:delta_length[i] * size + t]
        return pos_emb

    def get_state_pos_emb(self, t):
        size = self.config.state_size
        if self.config.emb_mode is not None:
            length = t // size + (1 if t % size != 0 else 0)
            pos_emb = self.state_s_emb.view(1, 1, size, -1).repeat(1, length, 1, 1)
            t_emb = self.t_emb[:, :length].view(1, length, 1, -1)
            pos_emb = pos_emb + t_emb
            pos_emb = pos_emb.view(1, length * size, -1)
            pos_emb = pos_emb[:, :t, :]
        else:
            pos_emb = self.state_pos_emb[:, :t]
        return pos_emb

    def forward(self, idx, cond_idx=torch.tensor([]), state_idx=torch.tensor([]), lbl_idx=torch.tensor([]), delta_length_cond=None):
        # prepapre frame embeddings
        tok_emb = self.tok_emb(idx)
        pos_emb = self.get_pos_emb(tok_emb.shape[1])
        emb = tok_emb + pos_emb


        # prepare cond embeddings
        use_cond = 0 not in cond_idx.size()
        if use_cond:
            cond_tok_emb = self.tok_emb(cond_idx)
            cond_pos_emb = self.get_pos_emb(cond_tok_emb.shape[1], delta_length_cond)
            cond_emb = cond_tok_emb + cond_pos_emb

        # prepare state embeddings
        use_state = 0 not in state_idx.size()
        if use_state:
            state_idx = state_idx[:, :self.config.num_blocks * self.config.state_size]
            state_tok_emb = self.state_tok_emb(state_idx)
            state_pos_emb = self.get_state_pos_emb(state_tok_emb.shape[1])
            state_emb = state_tok_emb + state_pos_emb

        # get num of tokens
        t = emb.shape[1]
        # t = emb.shape[1] + (state_emb.shape[1] if use_state else 0)
        t_cond = cond_emb.shape[1] if use_cond else 0

        # merge with state
        if use_state:
            if self.config.state_front:
                # put all frames annotation up front and then all frames
                emb = torch.cat([state_emb, emb], dim=1)
            else:
                # intertwine states and frames
                height, width = self.config.shape
                size = height * width
                length = emb.shape[1] // size
                state_size = self.config.state_size

                if length > 0:
                    state_emb0 = state_emb[:, :length * state_size].view(state_emb.size(0), length, state_size, -1)
                    emb0 = emb[:, :length * size].view(emb.size(0), length, size, -1)
                    emb0 = torch.cat([state_emb0, emb0], dim=2).view(emb.size(0), length * (size + state_size), -1)

                    state_emb1 = state_emb[:, length * state_size:(length + 1) * state_size]
                    emb1 = emb[:, length * size:]
                    emb1 = torch.cat([state_emb1, emb1], dim=1)

                    emb = torch.cat([emb0, emb1], dim=1)
                else:
                    emb = state_emb[:, :state_size].view(state_emb.size(0), state_size, -1)


        # merge with cond
        if use_cond:
            emb = torch.cat([cond_emb, emb], dim=1)

        if self.config.use_start_token:
            start_emb = self.start_tok_emb.repeat(emb.size(0), 1)
            emb = torch.cat([start_emb.unsqueeze(1), emb], dim=1)
            t += 1

        if self.config.use_lbl:
            lbl_emb = self.lbl_emb(lbl_idx)
            emb = torch.cat([lbl_emb.unsqueeze(1), emb], dim=1)
            t += 1

        assert t_cond + t <= self.block_size, "Cannot forward, model block size is exhausted."
        x = self.drop(emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)[:, t_cond:]

        return logits


class CGPT(nn.Module):
    """  the full GPT language model, with a context size of block_size, adapted to make n proposals  """

    def __init__(self, n_proposals, block_size, n_layer=12, n_head=8, n_embd=256, embd_pdrop=0., resid_pdrop=0.,
                 attn_pdrop=0., n_unmasked=0, n_in=3, resid_noise=False):
        super().__init__()
        config = GPTConfig(block_size=block_size, n_proposals=n_proposals, embd_pdrop=embd_pdrop,
                           resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop, n_layer=n_layer, n_head=n_head,
                           n_embd=n_embd, n_unmasked=n_unmasked, n_in=n_in, resid_noise=resid_noise)
        # input embedding stem
        self.tok_emb = nn.Linear(config.n_in, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        logits_size = config.n_proposals if config.n_proposals > 1 else 0
        self.head = nn.Linear(config.n_embd, config.n_proposals * config.n_in + logits_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config


    def get_block_size(self):
        return self.block_size


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, token_embeddings, single=False, lbl_idx=None):
        token_embeddings = self.tok_emb(token_embeddings)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        if single:
            x = x[:, [-1]]
        pred = self.head(x)
        if self.config.n_proposals > 1:
            pred = pred.view(-1, t, self.config.n_proposals, self.config.n_in + 1)
            logits, proposals = pred[:, :, :, 0], pred[:, :, :, 1:]
            return logits, proposals
        return pred


class DummyGPT(nn.Module):
    # for debugging
    def __init__(self, add_value=1):
        super().__init__()
        self.add_value = add_value

    def forward(self, idx):
        return idx + self.add_value, None


class CodeGPT(nn.Module):
    """Takes in semi-embeddings"""
    def __init__(self, vocab_size, block_size, in_channels, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        # input embedding stem
        self.tok_emb = nn.Linear(in_channels, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector

        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss



#### sampling utils

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x



#### clustering utils

class KMeans(nn.Module):
    def __init__(self, ncluster=512, nc=3, niter=10):
        super().__init__()
        self.ncluster = ncluster
        self.nc = nc
        self.niter = niter
        self.shape = (3,32,32)
        self.register_buffer("C", torch.zeros(self.ncluster,nc))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def is_initialized(self):
        return self.initialized.item() == 1

    @torch.no_grad()
    def initialize(self, x):
        N, D = x.shape
        assert D == self.nc, D
        c = x[torch.randperm(N)[:self.ncluster]] # init clusters at random
        for i in range(self.niter):
            # assign all pixels to the closest codebook element
            a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
            # move each codebook element to be the mean of the pixels that assigned to it
            c = torch.stack([x[a==k].mean(0) for k in range(self.ncluster)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            print('done step %d/%d, re-initialized %d dead clusters' % (i+1, self.niter, ndead))
            c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters

        self.C.copy_(c)
        self.initialized.fill_(1)


    def forward(self, x, reverse=False, shape=None):
        if not reverse:
            # flatten
            bs,c,h,w = x.shape
            assert c == self.nc
            x = x.reshape(bs,c,h*w,1)
            C = self.C.permute(1,0)
            C = C.reshape(1,c,1,self.ncluster)
            a = ((x-C)**2).sum(1).argmin(-1) # bs, h*w indices
            return a
        else:
            # flatten
            bs, HW = x.shape
            """
            c = self.C.reshape( 1, self.nc,  1, self.ncluster)
            c = c[bs*[0],:,:,:]
            c = c[:,:,HW*[0],:]
            x =      x.reshape(bs,       1, HW,             1)
            x = x[:,3*[0],:,:]
            x = torch.gather(c, dim=3, index=x)
            """
            x = self.C[x]
            x = x.permute(0,2,1)
            shape = shape if shape is not None else self.shape
            x = x.reshape(bs, *shape)

            return x
