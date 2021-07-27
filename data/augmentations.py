import torch
import torch.nn.functional as F
import numpy as np
import random
import torchvision.transforms as transforms
import math

from scipy.ndimage.filters import gaussian_filter

# Adapted from https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation

def get_backwarp_grid(height, width):
    horizontal = torch.linspace(-1.0 + (1.0 / width), 1.0 - (1.0 / width), width).view(1, 1, 1, -1).expand(-1, -1, height, -1)
    vertical = torch.linspace(-1.0 + (1.0 / height), 1.0 - (1.0 / height), height).view(1, 1, -1, 1).expand(-1, -1, -1, width)
    return torch.cat([horizontal, vertical], dim=1).float()

def backwarp(input, flow, backwarp_grid, padding_value=0, mode='bilinear'):
    flow = torch.cat([flow[:, [0], :, :] / ((input.shape[3] - 1.0) / 2.0), flow[:, [1], :, :] / ((input.shape[2] - 1.0) / 2.0)], dim=1)
    return torch.nn.functional.grid_sample(input=input - padding_value, grid=(backwarp_grid + flow).permute(0, 2, 3, 1), mode=mode, padding_mode='zeros', align_corners=False) + padding_value

def get_zoom_flow(zoom, height, width, adapt_to_scale=True):
    if zoom >= 1 and adapt_to_scale:
        tgt_height = height / zoom
        tgt_width = width / zoom
    else:
        tgt_height = zoom * height
        tgt_width = zoom * width
    delta_height = height - tgt_height
    delta_width = width - tgt_width
    zoom_dx = delta_width / 2 - torch.arange(width) * delta_width / (width - 1)
    zoom_dy = delta_height / 2 - torch.arange(height) * delta_height / (height - 1)
    return zoom_dx, zoom_dy

def get_augmentation(img, backwarp_grid, dim, opt, layout=None):
    alpha = opt.elastic_alpha
    sigma = opt.elastic_sigma
    min_zoom = opt.elastic_min_zoom
    max_zoom = opt.elastic_max_zoom
    corruption = opt.elastic_corruption
    mean_corruption = opt.elastic_mean_corruption
    blur = opt.blur_first
    invert = opt.distort_first

    random_state = np.random.RandomState(None)
    shape = img.shape[-2:]
    alpha = alpha * shape[0]
    sigma = sigma * shape[0]

    # elastic transformation
    dx = torch.tensor(gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha)
    dy = torch.tensor(gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha)
    i_dx = None
    i_dy = None
    if invert:
        elastic_flow = torch.stack([dx, dy]).float()
        inv_elastic_flow = approx_flow_inversion(elastic_flow)
        i_dx = inv_elastic_flow[0]  # approximated inverse
        i_dy = inv_elastic_flow[1]  # approximated inverse

    # zooming transformation
    o_dx = None
    o_dy = None
    height, width = shape
    zoom = min_zoom + np.random.rand() * (max_zoom - min_zoom)
    zoom_dx, zoom_dy = get_zoom_flow(zoom, height, width)
    if invert:
        if zoom < 1:
            i_dx += zoom_dx.view(1, -1) # exact inverse
            i_dy += zoom_dy.view(-1, 1) # exact inverse
            o_dx = zoom_dx.view(1, -1).repeat(height, 1)
            o_dy = zoom_dy.view(-1, 1).repeat(1, width)
        else:
            dx += zoom_dx.view(1, -1)
            dy += zoom_dy.view(-1, 1)
            i_zoom_dx, i_zoom_dy = get_zoom_flow(1/zoom, height, width, adapt_to_scale=False)
            i_dx -= i_zoom_dx.view(1, -1) # exact inverse
            i_dy -= i_zoom_dy.view(-1, 1) # exact inverse
    else:
        if zoom < 1:
            dx += zoom_dx.view(1, -1)
            dy += zoom_dy.view(-1, 1)
        else:
            o_dx = zoom_dx.view(1, -1).repeat(height, 1)
            o_dy = zoom_dy.view(-1, 1).repeat(1, width)

    # create context and distorted image
    if invert:
        context_flow = torch.stack([dx, dy]).unsqueeze(0).float()
        context_img = backwarp(img.unsqueeze(0), context_flow, backwarp_grid)
        if o_dx is not None:
            other_flow = torch.stack([o_dx, o_dy]).unsqueeze(0).float()
            distorted_img = backwarp(img.unsqueeze(0), other_flow, backwarp_grid)
        else:
            distorted_img = img.unsqueeze(0).clone()
        flow = torch.stack([i_dx, i_dy]).unsqueeze(0).float()
    else:
        distorted_flow = torch.stack([dx, dy]).unsqueeze(0).float()
        distorted_img = backwarp(img.unsqueeze(0), distorted_flow, backwarp_grid)
        if o_dx is not None:
            other_flow = torch.stack([o_dx, o_dy]).unsqueeze(0).float()
            context_img = backwarp(img.unsqueeze(0), other_flow, backwarp_grid)
            flow = torch.stack([dx - o_dx, dy - o_dy]).unsqueeze(0).float()
        else:
            context_img = img.unsqueeze(0)
            flow = torch.stack([dx, dy]).unsqueeze(0).float()

    # create context and distorted layout
    if layout is not None:
        layout = layout.unsqueeze(0).float()
        if invert:
            context_flow = torch.stack([dx, dy]).unsqueeze(0).float()
            context_layout = backwarp(layout.unsqueeze(0), context_flow, backwarp_grid, mode='nearest')
            if o_dx is not None:
                other_flow = torch.stack([o_dx, o_dy]).unsqueeze(0).float()
                distorted_layout = backwarp(layout.unsqueeze(0), other_flow, backwarp_grid, mode='nearest')
            else:
                distorted_layout = layout.unsqueeze(0).clone()
            flow = torch.stack([i_dx, i_dy]).unsqueeze(0).float()
        else:
            distorted_flow = torch.stack([dx, dy]).unsqueeze(0).float()
            distorted_layout = backwarp(layout.unsqueeze(0), distorted_flow, backwarp_grid, mode='nearest')
            if o_dx is not None:
                other_flow = torch.stack([o_dx, o_dy]).unsqueeze(0).float()
                context_layout = backwarp(layout.unsqueeze(0), other_flow, backwarp_grid, mode='nearest')
                flow = torch.stack([dx - o_dx, dy - o_dy]).unsqueeze(0).float()
            else:
                context_layout = layout.unsqueeze(0)
                flow = torch.stack([dx, dy]).unsqueeze(0).float()

    # rescale image
    f = None
    if dim != shape[0]:
        f = dim / shape[0]
        tgt_shape = [dim, int(shape[1] * dim / shape[0])]
        distorted_img = F.interpolate(distorted_img, size=tgt_shape, mode='bilinear')
        context_img = F.interpolate(context_img, size=tgt_shape, mode='bilinear')
    else:
        tgt_shape = shape

    # rescale layout
    if layout is not None:
        if dim != shape[0]:
            tgt_shape = [dim, int(shape[1] * dim / shape[0])]
            distorted_layout = F.interpolate(distorted_layout.float(), size=tgt_shape, mode='nearest')
            context_layout = F.interpolate(context_layout.float(), size=tgt_shape, mode='nearest')
        else:
            tgt_shape = shape

    # reshape layout
    if layout is not None:
        distorted_layout = distorted_layout.squeeze(1).long()
        context_layout = context_layout.squeeze(1).long()
    else:
        distorted_layout, context_layout = torch.tensor([]), torch.tensor([])

    # apply blur
    if blur is not None:
        s1, s2 = blur
        s = s1 + (s2 - s1) * random.random()
        k = int(3 * s) + 1 if int(3 * s) % 2 == 0 else int(3 * s)
        t = transforms.GaussianBlur(kernel_size=max(3, min(k, 13)), sigma=s)
        context_img = t(context_img)

    # apply corruption
    if corruption:
        corr_level = 1 - 2 * mean_corruption
        corr_mask = torch.tensor(gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha) > corr_level
        mask = backwarp(corr_mask.view(1, 1, *shape).float(), flow, backwarp_grid, padding_value=1)
        corr_mask = F.interpolate(corr_mask.view(1, 1, *shape).float(), size=tgt_shape, mode='bilinear')
        context_img = context_img * (1 - corr_mask).unsqueeze(0)
        mask = F.interpolate(mask, size=tgt_shape, mode='bilinear') > 0.5
    else:
        mask = torch.tensor([])

    # rescale flow
    if f is not None:
        flow = F.interpolate(flow * f, size=tgt_shape, mode='bilinear')

    return context_img.squeeze(0), context_layout.squeeze(0), distorted_img.squeeze(0), distorted_layout.squeeze(0), flow.squeeze(0), mask

def approx_flow_inversion(input, k=3):
    height, width = input.shape[1:]
    x_grid = torch.arange(width).view(1, -1).repeat(height, 1).view(-1).float()
    y_grid = torch.arange(height).view(-1, 1).repeat(1, width).view(-1).float()
    dx = input[0].view(-1)
    dy = input[1].view(-1)
    y_grid += dy
    x_grid += dx
    y_grid[y_grid < 0] = 0
    x_grid[x_grid < 0] = 0
    y_grid[y_grid > height - 1] = 0
    x_grid[x_grid > width - 1] = 0
    y_grid = y_grid.long()
    x_grid = x_grid.long()
    field = y_grid * width + x_grid
    inv_dx = torch.zeros_like(dx).scatter_(0, field, -dx).view(height, width)
    inv_dy = torch.zeros_like(dy).scatter_(0, field, -dy).view(height, width)
    mask = torch.zeros_like(dx).scatter_(0, field, 1).view(height, width).bool()

    padding = k // 2
    kernel = get_gaussian_kernel(k).view(1, 1, k, k)

    # fill missing value
    while not mask.all():
        # propagate mask
        new_mask = torch.zeros_like(mask)
        new_mask[1:] = (~mask[1:] & mask[:-1])
        new_mask[:-1] = (~mask[:-1] & mask[1:]) | new_mask[:-1]
        new_mask[:, 1:] = (~mask[:, 1:] & mask[:, :-1]) | new_mask[:, 1:]
        new_mask[:, :-1] = (~mask[:, :-1] & mask[:, 1:]) | new_mask[:, :-1]
        # compute missing values using kxk mean
        new_inv_dx = F.conv2d(inv_dx.view(1, 1, height, width), kernel, padding=padding).view(height, width)
        new_inv_dy = F.conv2d(inv_dy.view(1, 1, height, width), kernel, padding=padding).view(height, width)
        new_sum = F.conv2d(mask.float().view(1, 1, height, width), kernel, padding=padding).view(height, width)
        inv_dx[new_mask] = new_inv_dx[new_mask] / new_sum[new_mask]
        inv_dy[new_mask] = new_inv_dy[new_mask] / new_sum[new_mask]
        # update mask
        mask = mask | new_mask

    return torch.stack([inv_dx, inv_dy])

def get_gaussian_kernel(k):
    x_cord = torch.arange(k)
    x_grid = x_cord.repeat(k).view(k, k)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (k - 1) / 2.
    sigma = k / 6
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel