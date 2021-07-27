import random
import PIL
import os
import numpy as np
from copy import deepcopy
import pickle as pkl

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets.video_utils import VideoClips

from tools.utils import get_vprint, serialize, deserialize
from data.augmentations import get_augmentation, get_backwarp_grid

class BaseDataset(data.Dataset):
    def __init__(self, opt, phase='train', fold=None, from_vid=False, load_vid=False, verbose=True):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.phase = phase
        self.from_vid = from_vid
        self.load_vid = load_vid

        vprint = get_vprint(verbose)

        data_path = self.get_path_to_serialized("data", phase, fold)

        if opt.load_data and os.path.exists(data_path):
            self.data = deserialize(data_path)
        else:
            self.data = self.get_data(opt, phase=phase, from_vid=from_vid)
        if opt.save_data:
            vprint(f"Saving dataset paths and labels to {data_path}")
            serialize(data_path, self.data)

        if "vid_labels" in self.data:
            cat_idx, n_cat = np.unique(self.data["vid_labels"], return_counts=True)
            summary = [f"{n_cat[idx]} '{opt.categories[i]}'" for idx, i in enumerate(cat_idx)]
            vprint("Found " + ", ".join(summary) + " video files")

        if from_vid:
            metadata_path = self.get_path_to_serialized("metadata", phase, fold)
            if os.path.exists(metadata_path) and not opt.force_compute_metadata:
                vprint(f"Loading dataset metadata from {metadata_path}")
                metadata = deserialize(metadata_path)
                vprint(f"Metadata loaded")
                if metadata["video_paths"] != self.data["vid_paths"]:
                    vprint(f"Video paths have changed: initially {len(metadata['video_paths'])} files and now {len(self.data['vid_paths'])}")
                    vprint(f"Recomputing metadata")
                    metadata = None
            else:
                metadata = None

            if load_vid:
                frames_per_clip = self.opt.p2p_len if (self.opt.p2p_len is not None and self.phase == "train") else self.opt.vid_len
                frames_per_clip = self.opt.load_vid_len if (self.opt.load_vid_len is not None) else frames_per_clip #  and self.phase == "train"
            else:
                frames_per_clip = self.opt.img_out_of_n if self.phase == "train" else 1

            self.vid_clips = self.get_clips("vid", frames_per_clip, metadata=metadata)

            if metadata is None:
                vprint(f"Saving dataset metadata to {metadata_path}")
                serialize(metadata_path, self.vid_clips.metadata)

            self.dataset_size = self.vid_clips.num_clips()
        else:
            self.dataset_size = len(self.data['vid_frame_paths']) if (load_vid or self.opt.img_out_of_n > 1) else len(self.data['frame_paths'])

        self.dim_ind = int(np.log2(opt.dim)) - 2
        self.dim = [2 ** k for k in range(2, int(np.log2(opt.max_dim)) + 1)]
        if self.opt.load_elastic_view:
            dim = self.dim[self.dim_ind]
            self.backwarp_grid = get_backwarp_grid(dim, int(dim * opt.aspect_ratio))
            self.backwarp_grid_true = get_backwarp_grid(opt.hr_dim, int(opt.hr_dim * opt.aspect_ratio))

    def get_path_to_serialized(self, data_str, phase, fold):
        if self.opt.data_specs is None:
            path = os.path.join(self.opt.dataroot, f"{phase}_{data_str}.pkl")
        else:
            if fold is None:
                path = os.path.join(self.opt.dataroot, f"{self.opt.data_specs}_{phase}_{data_str}.pkl")
            else:
                path = os.path.join(self.opt.dataroot, f"folds/{self.opt.data_specs}_{fold}_{phase}_{data_str}.pkl")
        if fold is not None:
            assert os.path.exists(path)
        return path

    def get_data(self, opt, phase, from_vid):
        assert False, "A subclass of BaseDataset must override self.get_data(self, opt, phase, from_vid)"
        return {}

    def path_matches(self, vid_paths, vid2_paths):
        assert len(vid_paths) == len(vid2_paths)
        matches = [os.path.basename(vp) == os.path.basename(v2p) for vp, v2p in zip(vid_paths, vid2_paths)]
        if not all(matches):
            print(f"{np.sum(matches)}/{len(matches)} path matches")
        return all(matches)

    def get_clips(self, data_type, frames_per_clip, metadata=None):
        data = None
        if metadata is None:
            try:
                metadata = deepcopy(self.vid_clips.metadata)
                assert self.path_matches(metadata["video_paths"], self.data[f"{data_type}_paths"])
                metadata["video_paths"] = self.data[f"{data_type}_paths"]
            except:
                metadata = None
                data = self.data[f"{data_type}_paths"]
        return VideoClips(data,
                          clip_length_in_frames=frames_per_clip,
                          frames_between_clips=self.opt.vid_skip,
                          num_workers=self.opt.num_workers,
                          _precomputed_metadata=metadata)

    def get_augmentation_parameters(self):
        v_flip = random.random() > 0.5 if self.phase == 'train' and not self.opt.no_v_flip else False
        h_flip = random.random() > 0.5 if self.phase == 'train' and not self.opt.no_h_flip else False
        h = int(self.opt.true_dim)
        w = int(self.opt.true_dim * self.opt.true_ratio)
        if self.opt.fixed_top_centered_zoom:
            h_crop = int(h / self.opt.fixed_top_centered_zoom)
            w_crop = int(h_crop * self.opt.aspect_ratio)
            top_crop = 0
            assert w >= w_crop
            left_crop = int((w - w_crop) / 2)
            scale = None
        elif self.opt.fixed_crop:
            h_crop = self.opt.fixed_crop[0] # if self.phase == 'train' else h
            w_crop = self.opt.fixed_crop[1] # if self.phase == 'train' else w
            zoom = self.opt.min_zoom + random.random() * (self.opt.max_zoom - self.opt.min_zoom) if self.phase == 'train' else 1.
            h_scaled = int(h * zoom)
            w_scaled = int(w * zoom)
            scale = (h_scaled, w_scaled)
            assert h_scaled - h_crop >= 0
            assert w_scaled - w_crop >= 0
            h_p, w_p = (0.5, 0.5) if self.opt.centered_crop else (random.random(), random.random())
            top_crop = int(h_p * (h_scaled - h_crop)) # if self.phase == 'train' else 0
            left_crop = int(w_p * (w_scaled - w_crop)) # if self.phase == 'train' else 0
        else:
            min_zoom = max(1., self.opt.aspect_ratio / self.opt.true_ratio)
            max_zoom = max(self.opt.max_zoom, min_zoom)
            zoom = min_zoom + random.random() * (max_zoom - min_zoom) if self.phase == 'train' else min_zoom
            h_crop = int(h / zoom)
            w_crop = int(h_crop * self.opt.aspect_ratio)
            assert h >= h_crop
            assert w >= w_crop
            top_crop = int(random.random() * (h - h_crop)) if self.phase == 'train' else 0
            left_crop = int(random.random() * (w - w_crop)) if self.phase == 'train' else 0
            scale = None
        if self.opt.colorjitter is not None and self.phase == 'train':
            brightness = (random.random() * 2 - 1) * self.opt.colorjitter
            contrast = (random.random() * 2 - 1) * self.opt.colorjitter
            saturation = (random.random() * 2 - 1) * self.opt.colorjitter
            brightness = max(0, 1 + brightness)
            contrast = max(0, 1 + contrast)
            saturation = max(0, 1 + saturation)
            colorjitter = [[brightness, brightness], [contrast, contrast], [saturation, saturation]]
        else:
            colorjitter = None
        return v_flip, h_flip, top_crop, left_crop, h_crop, w_crop, scale, colorjitter

    def __getitem__(self, index):
        dim = self.dim[self.dim_ind]
        v_flip, h_flip, top_crop, left_crop, h_crop, w_crop, scale, colorjitter = self.get_augmentation_parameters()

        if self.from_vid and not self.load_vid:
            if self.phase == "train":
                img_idx = list(np.random.choice(self.opt.img_out_of_n, size=self.opt.n_consecutive_img, replace=False))
            else:
                img_idx = [0]
        else:
            img_idx = None

        transform_rgb = get_transform(dim, v_flip=v_flip, h_flip=h_flip, top_crop=top_crop, left_crop=left_crop,
                                      h_crop=h_crop, w_crop=w_crop, resize=self.opt.resize_img, scale=scale,
                                      imagenet=self.opt.imagenet_norm, colorjitter=colorjitter,
                                      is_PIL=not self.from_vid, resize_center_crop=self.opt.resize_center_crop_img,
                                      img_idx=img_idx)
        transform_seg = get_transform(dim, v_flip=v_flip, h_flip=h_flip, top_crop=top_crop, left_crop=left_crop,
                                      h_crop=h_crop, w_crop=w_crop, resize=self.opt.resize_img, scale=scale,
                                      is_PIL=not self.from_vid, resize_center_crop=self.opt.resize_center_crop_img,
                                      method=PIL.Image.NEAREST, img_idx=img_idx, normalize=False)
        transform_rgb_hr = get_transform(self.opt.hr_dim, v_flip=v_flip, h_flip=h_flip, resize=self.opt.resize_img,
                                          scale=scale, imagenet=self.opt.imagenet_norm,colorjitter=colorjitter,
                                          is_PIL=not self.from_vid, resize_center_crop=self.opt.resize_center_crop_img,
                                          img_idx=img_idx)
        transform_seg_hr = get_transform(self.opt.hr_dim, v_flip=v_flip, h_flip=h_flip, resize=self.opt.resize_img, scale=scale,
                                         is_PIL=not self.from_vid, resize_center_crop=self.opt.resize_center_crop_img,
                                         method=PIL.Image.NEAREST, img_idx=img_idx, normalize=False)

        load_rgb_path = lambda p: transform_rgb(PIL.Image.open(p).convert('RGB'))
        load_seg_path = lambda p: (transform_seg(PIL.Image.open(p))[0] * 255).long()
        load_rgb_hr_path = lambda p: transform_rgb_hr(PIL.Image.open(p).convert('RGB'))
        load_seg_hr_path = lambda p: (transform_seg_hr(PIL.Image.open(p))[0] * 255).long()

        input_dict = {}

        if self.from_vid:
            vid, _, _, video_index = self.vid_clips.get_clip(index)
            vid = (vid.float() / 255).permute(0, 3, 1, 2)
            if 'vid_labels' in self.data:
                input_dict['vid_lbl'] = self.data['vid_labels'][video_index]
            if 'vid_id' in self.data:
                input_dict['vid_id'] = self.data['vid_id'][video_index]
            if self.load_vid:
                if self.opt.load_vid_len is not None:
                    vid_len = self.opt.vid_len if self.opt.p2p_len is None else self.opt.p2p_len
                    step = min(max(1, int(random.random() * (self.opt.load_vid_len - 1) / (vid_len - 1))), self.opt.max_vid_step)
                    start = int(random.random() * (self.opt.load_vid_len - (vid_len - 1) * step)) if self.phase == "train" else 0
                    end = start + step * (vid_len - 1) + 1
                    vid = vid[start:end:step]
                if self.opt.p2p_len is not None:
                    idx = random.randrange(self.opt.p2p_len - self.opt.vid_len + 1)
                    idx_end_frame = random.randrange(idx + self.opt.vid_len - 1, self.opt.p2p_len)
                    vid = torch.cat([vid[idx:idx + self.opt.vid_len - 1], vid[[idx_end_frame]]], dim=0)
                    input_dict['delta_length'] = torch.tensor(idx_end_frame - idx)
                input_dict['vid'] = transform_rgb(vid)
                if 'stft_paths' in self.data:
                    if self.opt.load_vid_len is not None and self.opt.p2p_len is None:
                        stft_pickle = self.data['stft_paths'][video_index]
                        with open(stft_pickle, "rb") as f:
                            stft = pkl.load(f)
                        stft = stft[start:end:step].astype(np.float32)
                        stft = (torch.tensor(stft) * 2 - 1).unsqueeze(1)
                        stft = F.interpolate(stft, size=(64, 16), mode="bilinear")
                        input_dict['stft'] = stft
            else:
                img = transform_rgb(vid)
                if self.opt.load_elastic_view:
                    items = get_augmentation(img[0], self.backwarp_grid, dim, self.opt)
                    context_img, _, distorted_img, _, flow, mask = items
                    img[0] = context_img
                    input_dict['mask_img'] = mask
                    input_dict['flow_img'] = flow
                    img = torch.cat([img, distorted_img.unsqueeze(0)])
                input_dict['img'] = img.squeeze(0)
        else:
            if self.load_vid:
                vid_frame_path = self.data['vid_frame_paths'][index]
                vid_layout_path = self.data['vid_layout_paths'][index] if "vid_layout_paths" in self.data else None
                frames_per_clip = self.opt.p2p_len if (self.opt.p2p_len is not None and self.phase == "train") else self.opt.vid_len
                frames_per_clip = self.opt.load_vid_len if (self.opt.load_vid_len is not None and self.phase == "train") else frames_per_clip
                assert len(vid_frame_path) >= frames_per_clip, f"{vid_frame_path}, {frames_per_clip}"
                idx = random.randrange(len(vid_frame_path) - (frames_per_clip * self.opt.one_every_n) + 1)
                vid_frame_path = vid_frame_path[idx:idx + frames_per_clip * self.opt.one_every_n:self.opt.one_every_n]
                vid_layout_path = vid_layout_path[idx:idx + frames_per_clip * self.opt.one_every_n:self.opt.one_every_n] if vid_layout_path is not None else None
                if self.opt.load_vid_len is not None and self.phase == 'train':
                    vid_len = self.opt.vid_len if self.opt.p2p_len is None else self.opt.p2p_len
                    step = max(1, int(random.random() * (self.opt.load_vid_len - 1) / (vid_len - 1)))
                    start = int(random.random() * (self.opt.load_vid_len - (vid_len - 1) * step))
                    end = start + step * (vid_len - 1) + 1
                    vid_frame_path = vid_frame_path[start:end:step]
                    vid_layout_path = vid_layout_path[start:end:step] if vid_layout_path is not None else None
                if self.opt.p2p_len is not None and self.phase == 'train':
                    idx = random.randrange(self.opt.p2p_len - self.opt.vid_len + 1)
                    idx_end_frame = random.randrange(idx + self.opt.vid_len - 1, self.opt.p2p_len)
                    vid_frame_path = vid_frame_path[idx:idx + self.opt.vid_len - 1] + [vid_frame_path[idx_end_frame]]
                    vid_layout_path = vid_layout_path[idx:idx + self.opt.vid_len - 1] + [vid_layout_path[idx_end_frame]] if vid_layout_path is not None else None
                    input_dict['delta_length'] = torch.tensor(idx_end_frame - idx)
                vid = torch.zeros(self.opt.vid_len, 3, dim, dim * self.opt.aspect_ratio)
                for k, frame_path in enumerate(vid_frame_path):
                    vid[k] = load_rgb_path(frame_path)
                input_dict['vid'] = vid
                if vid_layout_path is not None:
                    layout = torch.zeros(self.opt.vid_len, dim, dim * self.opt.aspect_ratio)
                    for k, layout_path in enumerate(vid_layout_path):
                        layout[k] = load_seg_path(layout_path)
                    input_dict['layout'] = layout.long()
                if 'vid_labels' in self.data:
                    vid_lbl = self.data['vid_labels'][index]
                    input_dict['vid_lbl'] = vid_lbl
                if 'vid_frame_states' in self.data:
                    if self.opt.load_vid_len is None and self.opt.p2p_len is None:
                        states = self.data['vid_frame_states'][index]
                        states = states[idx: idx + frames_per_clip * self.opt.one_every_n:self.opt.one_every_n]
                        state = torch.tensor(states)
                        input_dict['state'] = state
            else:
                if self.opt.n_consecutive_img > 1 or self.opt.load_elastic_view:
                    vid_layout_path = None
                    if self.opt.n_consecutive_img == 1:
                        vid_frame_path = [self.data['frame_paths'][index]]
                    else:
                        vid_frame_path = self.data['vid_frame_paths'][index]
                        vid_layout_path = self.data['vid_layout_paths'][index] if "vid_layout_paths" in self.data else None
                        idx = random.randrange(len(vid_frame_path) - self.opt.img_out_of_n + 1)
                        vid_frame_path = vid_frame_path[idx:idx + self.opt.img_out_of_n]
                        img_idx = list(np.random.choice(self.opt.img_out_of_n, size=self.opt.n_consecutive_img, replace=False))
                        vid_frame_path = [vid_frame_path[i] for i in img_idx]
                        vid_layout_path = [vid_layout_path[i] for i in img_idx] if vid_layout_path is not None else None
                    img = torch.zeros(self.opt.n_consecutive_img, 3, dim, dim * self.opt.aspect_ratio)
                    for k, frame_path in enumerate(vid_frame_path):
                        if not (k == 0 and self.opt.load_elastic_view):
                            img[k] = load_rgb_path(frame_path)
                    if vid_layout_path is not None:
                        layout = torch.zeros(self.opt.n_consecutive_img, dim, dim * self.opt.aspect_ratio)
                        for k, layout_path in enumerate(vid_layout_path):
                            if not (k == 0 and self.opt.load_elastic_view):
                                layout[k] = load_seg_path(layout_path)
                    if self.opt.load_elastic_view:
                        hr_layout = load_seg_hr_path(vid_layout_path[0]) if vid_layout_path is not None else None
                        items = get_augmentation(load_rgb_hr_path(vid_frame_path[0]), self.backwarp_grid_true, dim, self.opt, layout=hr_layout)
                        context_img, context_layout, distorted_img, distorted_layout, flow, mask = items
                        img[0] = context_img
                        input_dict['mask_img'] = mask
                        input_dict['flow_img'] = flow
                        img = torch.cat([img, distorted_img.unsqueeze(0)])
                        if vid_layout_path is not None:
                            layout[0] = context_layout
                            layout = torch.cat([layout, distorted_layout.unsqueeze(0)])
                    input_dict['img'] = img
                    if vid_layout_path is not None:
                        input_dict['layout'] = layout.long().unsqueeze(1)
                    if 'vid_labels' in self.data:
                        vid_lbl = self.data['vid_labels'][index]
                        input_dict['vid_lbl'] = vid_lbl
                else:
                    frame_path = self.data['frame_paths'][index]
                    if 'vid_labels' in self.data:
                        frame_lbl = self.data['frame_labels'][index]
                        input_dict['vid_lbl'] = frame_lbl
                    if 'frame_states' in self.data:
                        input_dict['state'] = torch.tensor(self.data['frame_states'][index])
                    if not self.opt.no_rgb_img_from_img:
                        input_dict['img'] = load_rgb_path(frame_path)

        if self.opt.categories is not None:
            input_dict["tgt_vid_lbl"] = torch.randint(low=0, high=len(self.opt.categories), size=torch.Size([]))

        return input_dict

    def __len__(self):
        return self.dataset_size


def get_transform(dim, v_flip=False, h_flip=False, method=PIL.Image.BILINEAR, normalize=True, imagenet=False,
                  mask=False, flow=False, top_crop=None, left_crop=None, h_crop=None, w_crop=None, resize=None,
                  resize_center_crop=None, scale=None, colorjitter=None, is_PIL=True, img_idx=None,
                  blur=None):
    transform_list = []
    if img_idx is not None:
        transform_list.append(transforms.Lambda(lambda vid: vid[img_idx]))
    if resize is not None:
        transform_list.append(transforms.Resize(resize, method))
    if resize_center_crop is not None:
        transform_list.append(transforms.Resize(resize_center_crop, method))
        transform_list.append(transforms.CenterCrop(resize_center_crop))
    if scale is not None:
        transform_list.append(transforms.Resize(scale, method))
    if top_crop is not None:
        transform_list.append(transforms.Lambda(lambda img: transforms.functional.crop(img, top_crop, left_crop, h_crop, w_crop)))
    transform_list.append(transforms.Resize(dim, method))
    if v_flip:
        if is_PIL:
            transform_list.append(transforms.Lambda(lambda img: img.transpose(PIL.Image.FLIP_LEFT_RIGHT)))
        else:
            transform_list.append(transforms.Lambda(lambda img: img.flip(-1)))
    if h_flip:
        if is_PIL:
            transform_list.append(transforms.Lambda(lambda img: img.transpose(PIL.Image.FLIP_TOP_BOTTOM)))
        else:
            transform_list.append(transforms.Lambda(lambda img: img.flip(-2)))
    if colorjitter is not None:
        transform_list.append(transforms.ColorJitter(brightness=colorjitter[0], contrast=colorjitter[1], saturation=colorjitter[2]))
    if blur is not None:
        s1, s2 = blur
        s = s1 + (s2 - s1) * random.random()
        k = int(3 * s) + 1 if int(3 * s) % 2 == 0 else int(3 * s)
        transform_list.append(transforms.GaussianBlur(kernel_size=max(3, min(k, 13)), sigma=s))
    if is_PIL:
        transform_list.append(transforms.ToTensor())
    if normalize:
        if mask:
            transform_list.append(transforms.Normalize((0.5), (0.5)))
        elif flow:
            transform_list.append(transforms.Normalize((0.5, 0.5), (0.5, 0.5)))
        elif imagenet:
            transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        else:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)