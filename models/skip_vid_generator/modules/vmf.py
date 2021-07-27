# import torch
# import math
#
# # Adapted from https://github.com/Sachin19/seq2seq-con/blob/master/onmt/ive.py
# # and https://github.com/uclanlp/ELMO-C/blob/5a740fed6524d5d1dac0751ff0b359ae0c3730af/source/models/losses.py
#
# def nll_vMF(pred, tgt, lbd1=0.02, lbd2=0.1):
#     scale = pred.norm(p=2, dim=-1)
#     normalized_tgt = torch.nn.functional.normalize(tgt, p=2, dim=-1)
#     m = pred.size(-1)
#     norm_term = (m / 2 - 1) * torch.log(scale) - (m / 2) * math.log(2 * math.pi) - log_iv(m / 2 - 1, scale).float()
#     loss = - lbd2 * (pred * normalized_tgt).sum(-1) - norm_term + lbd1 * scale
#     return loss.mean()
#
# class LogIvFunctionWithUpperBoundGradientConstant(torch.autograd.Function):
#     @staticmethod
#     def forward(self, v, z):
#         self.save_for_backward(z)
#         self.v = v
#         return z - z
#
#     @staticmethod
#     def backward(self, grad_output):
#         z = self.saved_tensors[-1]
#         return None, ( grad_output.float() * (self.v / z + z / (self.v + torch.sqrt( (self.v+2) *(self.v+2) + z * z )) ) ).cuda()
#
# log_iv = LogIvFunctionWithUpperBoundGradientConstant.apply

# Adapted from https://github.com/Sachin19/seq2seq-con/blob/onmt-transformers/loss.py

import torch

def nll_vMF(pred, tgt):
    kappa = pred.norm(p=2, dim=-1)
    normalied_pred = torch.nn.functional.normalize(pred, p=2, dim=-1)
    normalized_tgt = torch.nn.functional.normalize(tgt, p=2, dim=-1)
    logcmk = Logcmk.apply
    loss = - logcmk(kappa, pred.size(-1)) + torch.log(1 + kappa) * (0.2 - (normalied_pred * normalized_tgt).sum(dim=-1))
    return loss.mean()

import scipy.special
import numpy as np
from torch.autograd import Variable

class Logcmk(torch.autograd.Function):
    """
    The exponentially scaled modified Bessel function of the first kind
    """
    @staticmethod
    def forward(ctx, k, m):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(k)
        ctx.m = m
        k = k.double()
        answer = (m/2-1)*torch.log(k) - torch.log(scipy.special.ive(m/2-1, k.cpu())).cuda() - k - (m/2)*np.log(2*np.pi)
        answer = answer.float()
        return answer

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        k, = ctx.saved_tensors
        m = ctx.m
        k = k.double()
        x = -((scipy.special.ive(m/2, k.cpu()))/(scipy.special.ive(m/2-1,k.cpu()))).cuda()
        x = x.float()
        return grad_output*Variable(x), None