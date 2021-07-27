# adapted from https://github.com/sthalles/SimCLR/blob/master/simclr.py
# and from https://github.com/HobbitLong/SupContrast/blob/master/losses.py


import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.temperature = opt.cont_temperature
        self.normalize = opt.cont_normalize

        if opt.cont_proj_size is not None:
            self.proj = nn.Sequential(nn.Linear(opt.style_size, opt.style_size, bias=False),
                                      nn.ReLU(),
                                      nn.Linear(opt.style_size, opt.cont_proj_size, bias=False))
        else:
            self.proj = lambda x: x

    def forward(self, x):
        """Compute contrastive loss from https://arxiv.org/pdf/2004.11362.pdf
        Bring closer features from the same sequence of frames and push further away the others.

        Args:
            x: hidden vectors of shape [batch_size, frames, vector_dim]

        Returns:
            A loss scalar.
        """
        x = self.proj(x)
        b, t, d = x.shape
        x = x.view(-1, d) # b*t d
        if self.normalize:
            x = F.normalize(x, dim=1)

        labels = torch.cat([i * torch.ones(t) for i in range(b)], dim=0).cuda() # b*t
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() # b*t b*t

        similarity_matrix = torch.div(torch.matmul(x, x.transpose(0, 1)), self.temperature)  # b*t b*t
        # for numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True) # b*t 1
        logits = similarity_matrix - logits_max.detach() # b*t b*t*
        # logits = similarity_matrix

        # discard the main diagonal from both: labels and logits
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda() # b*t b*t
        labels = labels[~mask].view(labels.shape[0], -1) # b*t b*t-1
        logits = logits[~mask].view(labels.shape[0], -1) # b*t b*t-1

        # compute log_prob
        exp_logits = torch.exp(logits) # b*t b*t-1
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # b*t b*t-1

        # compute mean of log-likelihood over positive
        sum_pos = t - 1
        mean_log_prob_pos = log_prob[labels.bool()].view(labels.shape[0], -1).sum(-1) / sum_pos # b*t

        # loss
        loss = - mean_log_prob_pos.mean()

        return loss