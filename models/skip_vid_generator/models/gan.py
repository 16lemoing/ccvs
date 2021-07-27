# Adapted from https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.skip_vid_generator.modules import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from tools.utils import flatten_vid, unflatten_vid

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


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
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualConv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            mul_kernel = kernel_size ** 3
            kernel_size = [kernel_size, kernel_size, kernel_size]
        else:
            mul_kernel = torch.prod(torch.tensor(kernel_size))

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, *kernel_size))
        self.scale = 1 / math.sqrt(in_channel * mul_kernel)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv3d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


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


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        use_style=True,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.use_style = use_style

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))

        if self.use_style:
            self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
            self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if self.use_style:
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
            weight = self.scale * self.weight * style
            if self.demodulate:
                demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
                weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        else:
            weight = (self.scale * self.weight).repeat(batch, 1, 1, 1, 1)

        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        use_style=True,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            use_style=use_style
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], use_style=True):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False, use_style=use_style)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ConvLayer3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, blur_kernel=[1, 3, 3, 1], bias=True,
                 activate=True, reduce_t=False):
        super().__init__()

        k = kernel_size if isinstance(kernel_size, int) else kernel_size[-1]
        k_t = kernel_size if isinstance(kernel_size, int) else kernel_size[0]

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (k - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

            stride = [1, 2, 2]
            self.padding = 0 if reduce_t else [k_t // 2, 0, 0]

        else:
            self.blur = None
            stride = 1
            self.padding = k // 2

        layers = []

        layers.append(
            EqualConv3d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
         if self.blur is not None:
             out = self.blur(input.view(input.size(0), -1, *input.shape[-2:]))
             out = out.view(*input.shape[:3], *out.shape[2:])
         else:
             out = input
         return self.layers(out)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], downsample=True):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample)
        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        return out


class ResBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], reduce_t=False):
        super().__init__()
        self.conv1 = ConvLayer3D(in_channel, in_channel, 3)
        self.conv2 = ConvLayer3D(in_channel, out_channel, 3, downsample=True, reduce_t=reduce_t)
        self.reduce_t = reduce_t
        kernel_skip = [3, 1, 1] if reduce_t else 1
        self.skip = ConvLayer3D(in_channel, out_channel, kernel_skip, downsample=True, activate=False, bias=False, reduce_t=reduce_t)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        return out


class StyleGAN2Discriminator(nn.Module):
    def __init__(self, opt, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        ndcf_mult = opt.ndcf_mult
        ndcf = opt.ndcf
        init_res = int(math.log(opt.z_shape[-2], 2)) - opt.downsample_dis_num
        final_res = init_res + len(ndcf_mult) - 1
        self.aspect_ratio = opt.aspect_ratio
        self.num_resolutions = len(ndcf_mult)
        self.n_consecutive_dis = opt.n_consecutive_dis
        self.downsample_dis_num = opt.downsample_dis_num
        if self.downsample_dis_num > 0:
            self.downsample = nn.AvgPool2d(2)

        block_in = ndcf * ndcf_mult[0]
        img_dim = 3 * self.n_consecutive_dis
        convs = [ConvLayer(img_dim, block_in, 1)]

        block_out = block_in
        for i in range(1, final_res - 1):
            if i < len(ndcf_mult):
                block_out = ndcf * ndcf_mult[i]

            convs.append(ResBlock(block_in, block_out, blur_kernel))

            block_in = block_out

        self.convs = nn.Sequential(*convs)

        self.stddev_group = opt.stddev_group
        self.stddev_feat = 1

        self.final_conv = ConvLayer(block_in + 1, block_in, 3)
        self.final_linear = nn.Sequential(
            EqualLinear(block_in * 4 * int(opt.aspect_ratio * 4), block_in, activation="fused_lrelu"),
            EqualLinear(block_in, 1),
        )

    def forward(self, input):
        if self.n_consecutive_dis > 1:
            n = self.n_consecutive_dis
            input = input.view(input.size(0) // n, n, *input.shape[1:]).contiguous()
            input = input.view(input.size(0), -1, *input.shape[3:])
        for _ in range(self.downsample_dis_num):
            input = self.downsample(input)
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = self.stddev_group
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return {"score": out}


class FeatureDiscriminator(nn.Module):
    def __init__(self, opt, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        in_size = opt.z_size
        convs = [ConvLayer(in_size, 128, 1)]

        h, w = opt.z_shape
        while h > 1 and w > 1:
            convs.append(ResBlock(128, 128, blur_kernel))
            h //= 2
            w //= 2

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(129, 128, 3)
        self.final_linear = nn.Sequential(
            EqualLinear(128 * h * w, 128, activation="fused_lrelu"),
            EqualLinear(128, 1),
        )

    def forward(self, input):
        input, vid_size = flatten_vid(input)

        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(batch, -1)
        out = self.final_linear(out)

        return {"score": out}


class StyleGAN2VidDiscriminator(nn.Module):
    def __init__(self, opt, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        ndcf_mult = opt.ndcf_mult
        ndcf = opt.ndcf #// 2
        init_res = int(math.log(opt.z_shape[-2], 2)) - opt.downsample_vdis_num
        final_res = init_res + len(ndcf_mult) - 1
        self.aspect_ratio = opt.aspect_ratio
        self.num_resolutions = len(ndcf_mult)
        self.downsample_vdis_num = opt.downsample_vdis_num
        if self.downsample_vdis_num > 0:
            self.downsample = nn.AvgPool2d(2)

        block_in = ndcf * ndcf_mult[0]
        convs = [ConvLayer3D(3, block_in, 1)]
        len_t = opt.vid_len

        block_out = block_in
        for i in range(1, final_res - 1):
            if i < len(ndcf_mult):
                block_out = ndcf * ndcf_mult[i]

            reduce_t = len_t > 2

            convs.append(ResBlock3D(block_in, block_out, blur_kernel, reduce_t=reduce_t))

            if reduce_t:
                len_t -= 2

            block_in = block_out

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer3D(block_in + 1, block_in, 3)
        self.final_linear = nn.Sequential(
            EqualLinear(block_in * 4 * int(4 * opt.aspect_ratio) * len_t, block_in, activation="fused_lrelu"),
            EqualLinear(block_in, 1),
        )

    def forward(self, input):
        if self.downsample_vdis_num > 0:
            bs, t = input.shape[:2]
            input = input.view(-1, *input.shape[2:])
            for _ in range(self.downsample_vdis_num):
                input = self.downsample(input)
            input = input.view(bs, t, *input.shape[1:])
        out = input.transpose(1, 2) # swap rgb and temporal dimensions
        out = self.convs(out)

        batch, channel, temp, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, temp, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4, 5], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, temp, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return {"score": out}

