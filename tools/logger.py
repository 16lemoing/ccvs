import numpy as np
from matplotlib import cm
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from tools.utils import color_transfer

class Logger():
    def __init__(self, opt):
        self.writer = SummaryWriter(opt.log_path)
        self.log_path = opt.log_path
        self.fps = opt.log_fps
        self.imagenet_norm = opt.imagenet_norm

    def is_empty(self, tensors):
        for tensor in tensors:
            if 0 in tensor.size():
                return True
        return False
    
    def log_img(self, name, tensor, nrow, global_iter, natural=True, normalize=False, span=None, pad_value=0):
        if self.is_empty([tensor]):
            return
        with torch.no_grad():
            tensor = tensor[:, :3]
            if natural and normalize and self.imagenet_norm:
                # tensor should be in [-1 1]
                tensor *= torch.tensor([[[[0.229]], [[0.224]], [[0.225]]]])
                tensor += torch.tensor([[[[0.485]], [[0.456]], [[0.406]]]])
                tensor = tensor.clamp(0, 1)
                normalize = False
            grid = make_grid(tensor, nrow=nrow, normalize=normalize, range=span, pad_value=pad_value)
            self.writer.add_image(name, grid, global_iter)

    def log_seg(self, name, tensor, seg_dim, nrow, global_iter):
        if self.is_empty([tensor]):
            return
        with torch.no_grad():
            if tensor.ndim == 4:
                seg = tensor.max(1, keepdim=True)[1]
            else:
                seg = tensor.unsqueeze(1)
            colormap = cm.get_cmap('hsv', seg_dim)(np.linspace(0, 1, seg_dim))
            seg = color_transfer(seg, colormap)
            self.log_img(name, seg, nrow, global_iter, normalize=True, span=(-1, 1))

    def log_vid(self, name, tensor, global_iter, natural=True, normalize=False, span=None, cond_frames=None):
        if self.is_empty([tensor]):
            return
        with torch.no_grad():
            tensor = tensor[:, :, :3]
            if natural and normalize and self.imagenet_norm:
                # tensor should be in [-1 1]
                tensor *= torch.tensor([[[[0.229]], [[0.224]], [[0.225]]]])
                tensor += torch.tensor([[[[0.485]], [[0.456]], [[0.406]]]])
                tensor = tensor.clamp(0, 1)
            elif normalize:
                tensor = tensor.clamp(span[0], span[1])
                tensor = (tensor - span[0]) / (span[1] - span[0])
            if cond_frames is not None:
                # show synthetic frames with red border
                low_h, low_w = int(0.025 * tensor.size(3)), int(0.025 * tensor.size(4))
                high_h, high_w = tensor.size(3) - low_h, tensor.size(4) - low_w
                red_color = torch.tensor([[[[1.]], [[0.]], [[0.]]]])
                tensor[:, cond_frames:, :, :low_h] = red_color
                tensor[:, cond_frames:, :, high_h:] = red_color
                tensor[:, cond_frames:, :, :, :low_w] = red_color
                tensor[:, cond_frames:, :, :, high_w:] = red_color
            self.writer.add_video(name, tensor, global_iter, self.fps)

    def log_flow(self, name, flow, nrow, global_iter):
        if self.is_empty([flow]):
            return
        with torch.no_grad():
            if len(flow.shape) == 5:
                bs, t, _, h, w = flow.shape
                flow = flow.permute(0, 1, 3, 4, 2)
                flow_vid = torch.zeros(bs, t, 3, h, w)
                for i in range(t):
                    flow_vid[:, i] = self.get_flow_rgb(flow[:, i])
                self.log_vid(name, flow_vid, global_iter)
            else:
                flow = flow.permute(0, 2, 3, 1)
                self.log_img(name, self.get_flow_rgb(flow), nrow, global_iter)
    
    def log_scalar(self, name, scalar, global_iter):
        if scalar is not None:
            if type(scalar) == list:
                for i, x in enumerate(scalar):
                    self.log_scalar(f"{name}_{i}", x, global_iter)
            else:
                self.writer.add_scalar(name, scalar, global_iter)

    def get_flow_rgb(self, flow, boost=10):
        r = (flow ** 2).sum(-1).sqrt() / np.sqrt(2) * boost
        r[r > 1] = 1.
        theta = (1 + torch.atan2(flow.select(-1, -1), flow.select(-1, 0)) / np.pi) / 2
        cmp = cm.get_cmap('hsv', 128)
        flow_rgba = cmp(theta.numpy())
        flow_rgb = torch.tensor(flow_rgba[:, :, :, :3]).float()
        flow_rgb = r.unsqueeze(-1) * flow_rgb
        return flow_rgb.permute(0, 3, 1, 2)