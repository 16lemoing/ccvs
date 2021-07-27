# Adapted from https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py

import math
import random

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import ops

from models.skip_vid_generator.modules import fused_leaky_relu, upfirdn2d, FunctionCorrelation
from tools.utils import flatten_vid, unflatten_vid

def cast(x, dtype):
    if x is not None:
        return x.to(dtype)
    return x

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer("kernel", kernel)
        self.pad = pad

    def forward(self, input):
        return upfirdn2d(input.float(), self.kernel, pad=self.pad).to(input.dtype)


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, transpose=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        self.transpose = transpose
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        if self.transpose:
            out = F.conv_transpose2d(input, self.weight.transpose(0, 1) * self.scale, bias=self.bias,
                                     stride=self.stride, padding=self.padding)
        else:
            out = F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return (f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
                f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})")


class ConvLayer(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, upsample=False, blur_kernel=[1, 3, 3, 1],
        bias=True, activate=True):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            stride = 2
            transpose = False
            self.padding = 0
        elif upsample:
            stride = 2
            transpose = True
            self.padding = 0
        else:
            stride = 1
            transpose = False
            self.padding = kernel_size // 2

        layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=self.padding, stride=stride,
                                  bias=bias, transpose=transpose)) # bias and not activate

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            layers.append(Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor))

        if activate:
            layers.append(nn.LeakyReLU(inplace=False, negative_slope=0.1))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], downsample=False, upsample=False):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample, upsample=upsample, blur_kernel=blur_kernel)
        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=downsample, upsample=upsample, blur_kernel=blur_kernel, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        return out


def get_backwarp_grid(height, width):
    horizontal = torch.linspace(-1.0 + (1.0 / width), 1.0 - (1.0 / width), width).view(1, 1, 1, -1).expand(-1, -1, height, -1)
    vertical = torch.linspace(-1.0 + (1.0 / height), 1.0 - (1.0 / height), height).view(1, 1, -1, 1).expand(-1, -1, -1, width)
    return torch.cat([horizontal, vertical], dim=1).cuda()


def backwarp(input, flow, backwarp_grid):
    flow = torch.cat([flow[:, [0], :, :] / ((input.shape[3] - 1.0) / 2.0), flow[:, [1], :, :] / ((input.shape[2] - 1.0) / 2.0)], dim=1)
    return torch.nn.functional.grid_sample(input=input, grid=(backwarp_grid + flow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)


class Matching(torch.nn.Module):
    def __init__(self, flow_mult, kernel, feat_size, use_corr, corr_stride, use_masked_flow, use_deformed_conv,
                 use_tradeoff, no_proj, first):
        super().__init__()
        self.flow_mult = flow_mult
        self.corr_stride = corr_stride
        self.use_corr = use_corr
        self.use_masked_flow = use_masked_flow
        self.use_deformed_conv = use_deformed_conv
        self.use_tradeoff = use_tradeoff

        if feat_size > 16 and not no_proj:
            tgt_size = max(16, feat_size // 4)
            self.proj = ConvLayer(feat_size, tgt_size, 1)
        else:
            self.proj = lambda x: x

        if first:
            self.upsample_flow = None
            self.upsample_occ = None
            self.upsample_toff = None
        else:
            self.upsample_flow = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1, bias=False, groups=2)
            self.upsample_occ = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False, groups=1)
            if self.use_tradeoff:
                self.upsample_toff = nn.ConvTranspose2d(in_channels=32, out_channels=feat_size, kernel_size=4, stride=2, padding=1, bias=False, groups=32)

        if self.use_deformed_conv:
            self.deform = ops.DeformConv2d(in_channels=feat_size, out_channels=feat_size, kernel_size=3, stride=1, padding=1)

        if self.use_deformed_conv or self.use_tradeoff:
            self.leaky_relu = nn.LeakyReLU(inplace=False, negative_slope=0.1)

        if self.use_corr:
            if corr_stride == 1:
                self.upsample_corr = None
            else:
                self.upsample_corr = nn.ConvTranspose2d(in_channels=49, out_channels=49, kernel_size=4, stride=2, padding=1, bias=False, groups=49)
            convs = [ConvLayer(49, 128, 3)]
        else:
            convs = [ConvLayer(feat_size * 2, 128, 3)]

        convs += [ConvLayer(128, 64, 3),
                  ConvLayer(64, 32, 3)]
        self.convs = torch.nn.Sequential(*convs)
        self.flow_head = ConvLayer(32, 2, kernel, activate=False)
        self.occ_head = ConvLayer(32, 1, kernel, activate=False)

    def forward(self, input, inter, flow, occ, toff, backwarp_grid):
        if flow is not None:
            flow = self.upsample_flow(flow)
            occ = self.upsample_occ(occ)
            if self.use_deformed_conv:
                b,_, h, w = flow.shape
                inter = self.deform(inter, (flow * self.flow_mult).unsqueeze(1).repeat_interleave(9, dim=1).view(b, -1, h, w))
            else:
                inter = backwarp(input=inter, flow=flow * self.flow_mult, backwarp_grid=backwarp_grid)
            if self.use_masked_flow:
                inter *= (1 - F.sigmoid(occ))
            if self.use_tradeoff:
                toff = self.upsample_toff(toff)
                inter += toff
            if self.use_deformed_conv or self.use_tradeoff:
                inter = self.leaky_relu(inter)

        if self.use_corr:
            corr = F.leaky_relu(FunctionCorrelation(self.proj(input).float(), self.proj(inter).float(), stride=self.corr_stride), negative_slope=0.1, inplace=False).to(input.dtype)
            if self.corr_stride != 1:
                corr = self.upsample_corr(corr)
            feat = self.convs(corr)
        else:
            feat = self.convs(torch.cat([input, inter], dim=1))

        flow = (flow if flow is not None else 0.0) + self.flow_head(feat)
        occ = (occ if occ is not None else 0.0) + self.occ_head(feat)
        return flow, occ


class Subpixel(torch.nn.Module):
    def __init__(self, flow_mult, kernel, feat_size, use_tradeoff):
        super().__init__()
        self.flow_mult = flow_mult
        self.use_tradeoff = use_tradeoff

        convs = [ConvLayer(2 * feat_size + 2 + 1, 128, 3),
                 ConvLayer(128, 64, 3),
                 ConvLayer(64, 32, 3)]
        self.convs = torch.nn.Sequential(*convs)
        self.flow_head = ConvLayer(32, 2, kernel, activate=False)
        self.occ_head = ConvLayer(32, 1, kernel, activate=False)

    def forward(self, input, inter, flow, occ, backwarp_grid):
        inter = backwarp(input=inter, flow=flow * self.flow_mult, backwarp_grid=backwarp_grid)
        feat = self.convs(torch.cat([input, inter, flow, occ], dim=1))
        flow = flow + self.flow_head(feat)
        occ = occ + self.occ_head(feat)
        toff = feat if self.use_tradeoff else None
        return flow, occ, toff


class InterBlock(nn.Module):
    def __init__(self, opt, height, width, flow_mult, kernel, feat_size, corr_stride, first=False):
        super().__init__()
        use_corr = not opt.no_corr
        use_masked_flow = opt.use_masked_flow
        use_deformed_conv = opt.use_deformed_conv
        use_tradeoff = opt.use_tradeoff
        no_proj = opt.no_proj
        self.flow_mult = flow_mult
        self.feat_size = feat_size
        self.backwarp_grid = get_backwarp_grid(int(height), int(width))
        self.matching = Matching(flow_mult, kernel, feat_size, use_corr, corr_stride, use_masked_flow,
                                 use_deformed_conv, use_tradeoff, no_proj, first=first)
        self.subpixel = Subpixel(flow_mult, kernel, feat_size, use_tradeoff)

    def forward(self, input, inters, flows=None, occs=None, toffs=None, eps=1e-6):
        dtype = input.dtype
        input = input.contiguous()
        k = len(inters)
        inters = torch.cat([inter.contiguous().unsqueeze(1) for inter in inters], dim=1).view(-1, *input.shape[1:])
        inputs = input.unsqueeze(1).repeat(1, k, 1, 1, 1).view(-1, *input.shape[1:])
        flows, occs = self.matching(inputs, inters, flows, occs, toffs, self.backwarp_grid)
        flows, occs, toffs = self.subpixel(inputs, inters, flows, occs, self.backwarp_grid)
        warped_inters = backwarp(input=inters, flow=flows * self.flow_mult, backwarp_grid=self.backwarp_grid)
        if k > 1:
            confs = (1 - F.sigmoid(occs)).view(-1, k, *occs.shape[1:]) + eps
            sum_confs = confs.sum(dim=1)
            warped_inter = (warped_inters.view(-1, k, *input.shape[1:]) * confs).sum(dim=1) / sum_confs
            occ = (occs.view(-1, k, *occs.shape[1:]) * confs).sum(dim=1) / sum_confs
        else:
            warped_inter = warped_inters
            occ = occs
        occ_mask = F.sigmoid(occ)
        input = (occ_mask * input + (1 - occ_mask) * warped_inter)
        return cast(input, dtype), cast(flows, dtype), cast(occs, dtype), cast(toffs, dtype)


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input.float(), self.kernel, up=self.factor, down=1, pad=self.pad).to(input.dtype)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ConvLayer(in_channel, 3, 1, activate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, skip=None):
        out = self.conv(input)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class SkipGANEncoder(nn.Module):
    def __init__(self, opt, blur_kernel=[1, 3, 3, 1], mode="rgb"):
        super().__init__()
        necf_mult = opt.necf_mult
        necf = opt.necf
        self.num_resolutions = len(necf_mult)
        self.z_size = opt.z_size
        self.normalize_out = opt.normalize_out
        self.mode =mode

        block_in = necf * necf_mult[0]
        in_size = opt.layout_size if mode == "layout" else 3
        blocks = [ConvLayer(in_size, block_in, 1)]

        inter_sizes = [int(opt.inter_p * block_in)]

        for i in range(1, self.num_resolutions):
            block_out = necf * necf_mult[i]
            blocks.append(ResBlock(block_in, block_out, blur_kernel, downsample=True))
            inter_sizes.append(int(opt.inter_p * block_out))
            block_in = block_out

        blocks.append(ConvLayer(block_out, self.z_size, 1))
        self.blocks = nn.ModuleList(blocks)

        self.inter_sizes = inter_sizes

    def forward(self, input):
        input, vid_size = flatten_vid(input)

        out = self.blocks[0](input)
        inter_enc = [out[:, :self.inter_sizes[0]]]

        for i in range(1, self.num_resolutions):
            out = self.blocks[i](out)
            inter_enc.append(out[:, :self.inter_sizes[i]])

        out = self.blocks[self.num_resolutions](out)

        if self.normalize_out:
            out = out / torch.norm(out, p=2, dim=1, keepdim=True)

        return unflatten_vid(out, vid_size), [unflatten_vid(feat, vid_size) for feat in inter_enc]


class SkipGANDecoder(nn.Module):
    def __init__(self, opt, blur_kernel=[1, 3, 3, 1], mode="rgb"):
        super().__init__()
        necf_mult = opt.necf_mult
        necf = opt.necf
        self.num_resolutions = len(necf_mult)
        self.use_inter = opt.use_inter
        self.z_size = opt.z_size
        self.skip_rgb = opt.skip_rgb
        self.skip_tanh = opt.skip_tanh
        self.mode = mode

        block_in = necf * necf_mult[-1]
        in_size = opt.z_size * 2 if mode == "both" else opt.z_size
        blocks = [ConvLayer(in_size, block_in, 1)]
        if self.skip_rgb:
            to_rgb = [ToRGB(block_in, upsample=False)]
        inter_sizes = [int(opt.inter_p * block_in)]
        for i in range(1, self.num_resolutions):
            block_out = necf * necf_mult[-1-i]
            blocks.append(ResBlock(block_in, block_out, blur_kernel, upsample=True))
            if self.skip_rgb:
                to_rgb.append(ToRGB(block_out))
            inter_sizes.append(int(opt.inter_p * block_out))
            block_in = block_out
        if self.skip_rgb:
            self.to_rgb = nn.ModuleList(to_rgb)
        else:
            if self.mode == "layout":
                blocks.append(ConvLayer(block_out, opt.layout_size, 1, activate=False))
            elif self.mode == "both":
                self.refine_layout = ConvLayer(block_out, block_out, 3)
                self.layout_head = ConvLayer(block_out, opt.layout_size, 1, activate=False)
                self.rgb_head = ConvLayer(block_out, 3, 1, activate=False)
            else:
                blocks.append(ConvLayer(block_out, 3, 1, activate=False))
        self.blocks = nn.ModuleList(blocks)

        self.backwarp_grid = None
        self.last_flow_mult = None
        if self.use_inter:
            inter_blocks = []
            height = opt.max_dim / (2 ** (self.num_resolutions - 1))
            width = int(height * opt.aspect_ratio)
            for i in range(self.num_resolutions):
                kernel = 2 ** (i // 2 + 1) + 1
                flow_mult = 2 ** i
                corr_stride = 2 if i > 2 else 1
                inter_blocks.append(InterBlock(opt, height, width, flow_mult, kernel, inter_sizes[i], corr_stride, first=i == 0))
                height *= 2
                width *= 2
            self.backwarp_grid = get_backwarp_grid(int(height / 2), int(width / 2))
            self.last_flow_mult = flow_mult
            self.inter_blocks = nn.ModuleList(inter_blocks)
            self.inter_sizes = inter_sizes

    def backwarp_img(self, input, flow):
        assert self.backwarp_grid is not None
        return backwarp(input, flow, self.backwarp_grid)

    def forward(self, input, inter_tgts=None, return_all=False, drop_p=0, inter_src=None, alpha_src=None,
                inter_pre_warping=True, has_ctx=True):
        input, vid_size = flatten_vid(input)
        inter_dec = []
        if inter_tgts is not None and self.use_inter:
            inter_tgts = [[flatten_vid(t)[0] for t in inter_tgt] for inter_tgt in inter_tgts]
        if inter_src is not None and self.use_inter:
            inter_src = [flatten_vid(t)[0] for t in inter_src]

        inter_idx = list(range(input.size(0)))
        if drop_p > 0:
            random.shuffle(inter_idx)
            inter_idx = inter_idx[:int((1 - drop_p) * input.size(0))]

        out = self.blocks[0](input)
        inter_flows, inter_occs = [], []
        if self.use_inter and has_ctx:
            s = self.inter_sizes[0]
            inter_dec.append(out[:, :s])
            if inter_src is not None:
                out[inter_idx, :s] = alpha_src[0] * inter_src[-1][inter_idx] + (1 - alpha_src[0]) * out[inter_idx, :s]
            inter_tgts_i = [inter_tgt[-1][inter_idx] for inter_tgt in inter_tgts]

            out[inter_idx, :s], flows, occs, toffs = self.inter_blocks[0](out[inter_idx, :s], inter_tgts_i)
            inter_flows.append(flows)
            inter_occs.append(occs)
        if self.skip_rgb:
            rgb = self.to_rgb[0](out)

        for i in range(1, self.num_resolutions):
            out = self.blocks[i](out)
            if self.use_inter and has_ctx:
                s = self.inter_sizes[i]
                if inter_pre_warping:
                    inter_dec.append(out[:, :s])
                if inter_src is not None:
                    out[inter_idx, :s] = alpha_src[i] * inter_src[-1-i][inter_idx] + (1 - alpha_src[i]) * out[inter_idx, :s]
                inter_tgts_i = [inter_tgt[-1-i][inter_idx] for inter_tgt in inter_tgts]
                out[inter_idx, :s], flows, occs, toffs = self.inter_blocks[i](out[inter_idx, :s], inter_tgts_i, flows, occs, toffs)
                if not inter_pre_warping:
                    inter_dec.append(out[:, :s])
                inter_flows.append(flows)
                inter_occs.append(occs)
                if self.skip_rgb:
                    rgb = self.to_rgb[i](out, rgb)

        out2 = torch.tensor([])
        if self.mode == "both":
            out1 = unflatten_vid(self.rgb_head(out), vid_size)
            out2 = unflatten_vid(self.layout_head(self.refine_layout(out)), vid_size)
        else:
            if self.skip_rgb:
                out1 = rgb
            else:
                out1 = self.blocks[self.num_resolutions](out)
            if self.skip_tanh:
                out1 = torch.tanh(out1)
            out1 = unflatten_vid(out1, vid_size)

        if return_all:
            inter_dec = [unflatten_vid(feat, vid_size) for feat in inter_dec]
            return out1, out2, inter_flows, inter_occs, inter_dec
        return out1, out2


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})")


class StateEstimator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        convs = []
        h, w = opt.z_shape
        in_size = opt.z_size
        while h > 1 and w > 1:
            convs.append(ConvLayer(in_size, opt.state_hsize, 3, downsample=True))
            h //= 2
            w //= 2
            in_size = opt.state_hsize
        self.convs = nn.Sequential(*convs)
        self.fc = EqualLinear(opt.state_hsize * h * w, opt.state_size)

    def forward(self, input):
        input, vid_size = flatten_vid(input)
        out = self.convs(input).view(input.size(0), -1)
        out = F.sigmoid(self.fc(out))
        return unflatten_vid(out, vid_size)

class StftEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        convs = [ConvLayer(1, opt.stft_hsize, 1, downsample=False)]
        for i in range(3):
            convs.append(ConvLayer(opt.stft_hsize, opt.stft_hsize, 3, downsample=True))
        convs.append(ConvLayer(opt.stft_hsize, opt.stft_size, 3, downsample=False))
        self.convs = nn.Sequential(*convs)

    def forward(self, input):
        input, vid_size = flatten_vid(input)
        out = self.convs(input)
        return unflatten_vid(out, vid_size)

class StftDecoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        convs = [ConvLayer(opt.stft_size, opt.stft_hsize, 3, upsample=False)]
        for i in range(3):
            convs.append(ConvLayer(opt.stft_hsize, opt.stft_hsize, 3, upsample=True))
        convs.append(ConvLayer(opt.stft_hsize, 1, 1, upsample=False))
        self.convs = nn.Sequential(*convs)

    def forward(self, input):
        input, vid_size = flatten_vid(input)
        out = F.tanh(self.convs(input))
        return unflatten_vid(out, vid_size)