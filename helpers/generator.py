import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from itertools import cycle

import warnings
warnings.filterwarnings("ignore")

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
code_dir = os.path.dirname(current_dir)
sys.path.insert(0, code_dir)

from tools.options import Options
from tools.engine import Engine
from tools.utils import mkdir, color_transfer
from models.skip_vid_generator.models.quantized_video_model import QVidModel
from models.skip_vid_generator.models.state_model import StateModel
from models.skip_vid_generator.models.transformer_model import Transformer
from models.skip_vid_generator.models.stft_model import StftModel

import time

class Generator:
    def __init__(self, opt):
        self.opt = opt["transformer"]
        self.qvid_opt = opt["qvid_generator"]
        self.state_opt = opt["state_estimator"]
        self.stft_ae_opt = opt["stft_ae"]
        self.iter_fn = cycle if self.opt.iter_function == "cycle" else iter

    def next_batch(self, data_info):
        try:
            return next(data_info["loader_iter"])
        except StopIteration:
            if data_info["num_folds"] is not None:
                epoch = data_info["epoch"] + 1 / data_info["num_folds"]
                fold = (data_info["fold"] + 1) % data_info["num_folds"]
                phase, data_type = data_info["phase"], data_info["data_type"]
                # free memory from previous fold before loading the next
                for k in data_info:
                    data_info[k] = None
                new_data_info = self.get_data_info(phase, data_type, fold)
                new_data_info["epoch"] = epoch
                for k in data_info:
                    data_info[k] = new_data_info[k]
            else:
                data_info["epoch"] += 1
                if self.engine.distributed:
                    data_info["datasampler"].set_epoch(int(data_info["epoch"]))
                data_info["loader_iter"] = self.iter_fn(data_info["dataloader"])
            return next(data_info["loader_iter"])

    @torch.no_grad()
    def generate_vid(self, data, global_iter):
        # downsize
        if self.opt.down_size is not None:
            vid = data["vid"]
            bs, t = vid.shape[:2]
            img = data["vid"].view(-1, *vid.shape[2:])
            img = F.interpolate(img, size=self.opt.down_size, mode='bilinear')
            img = F.interpolate(img, size=vid.shape[-2:], mode='bilinear')
            data["vid"] = img.view(bs, t, *vid.shape[2:])

        t0 = time.time()
        # encode all frames
        encoded_data = self.vid_model(data, mode=f'vid_encoder')
        t1 = time.time()

        # estimate state
        if self.opt.state:
            state_encoded_data = self.state_model(encoded_data, mode=f'vid_encoder')
            encoded_data.update(state_encoded_data)
            data.update(self.state_model(encoded_data, mode=f'vid_decoder'))
        if self.opt.stft:
            stft_encoded_data = self.stft_model(data, mode=f'vid_encoder')
            encoded_data.update(stft_encoded_data)

        # set code length
        if self.opt.p2p:
            cond_step = 1
            t_step = self.opt.vid_len - 1
        else:
            cond_step = 0
            t_step = self.opt.vid_len
        total_len = (cond_step + t_step) * torch.prod(torch.tensor(self.qvid_opt.z_shape))
        cond_len = cond_step * torch.prod(torch.tensor(self.qvid_opt.z_shape))
        if self.opt.state or self.opt.layout or self.opt.stft or self.opt.deblurring:
            total_len += t_step * self.opt.state_size

        # crop information
        if self.opt.gen_from_img:
            crop_prop = self.opt.cond_len / torch.prod(torch.tensor(self.qvid_opt.z_shape))
        else:
            crop_prop = self.opt.cond_len / (torch.prod(torch.tensor(self.qvid_opt.z_shape)) * self.opt.vid_len)
        cropped_encoded_data = {}
        cropped_encoded_data["code"] = encoded_data["code"][:, :int(crop_prop * encoded_data["code"].size(1))]
        encoded_data["inter"] = [feat[:, :int(crop_prop * feat.size(1))].contiguous() for feat in encoded_data["inter"]]
        cropped_encoded_data["inter"] = encoded_data["inter"]
        if self.opt.p2p:
            cropped_encoded_data["cond_code"] = encoded_data["code"][:, -self.opt.z_chunk:]
            cropped_encoded_data["cond_inter"] = [feat[:, -1:].contiguous() for feat in encoded_data["inter"]]
            cropped_encoded_data["delta_length_cond"] = torch.tensor([self.opt.vid_len - 1]).repeat(cropped_encoded_data["code"].size(0))
        if self.opt.state or self.opt.layout or self.opt.stft:
            if self.opt.keep_state:
                cropped_encoded_data["state_code"] = encoded_data["state_code"]
                if self.opt.layout:
                    cropped_encoded_data["interl"] = encoded_data["interl"]
            elif self.opt.custom_state:
                init_state = self.state_model(encoded_data, mode=f'vid_decoder')["state"][:, [0]]
                custom_state = square_trajectory(init_state, self.opt.vid_len)
                cropped_encoded_data["state_code"] = self.state_model(custom_state, mode=f'vid_encoder')["state_code"]
            else:
                cropped_encoded_data["state_code"] = encoded_data["state_code"][:, :int(crop_prop * encoded_data["state_code"].size(1))]
                if self.opt.layout:
                    cropped_encoded_data["interl"] = [feat[:, :int(crop_prop * feat.size(1))].contiguous() for feat in encoded_data["interl"]]
        if self.opt.cat:
            if "vid_lbl" not in data:
                data["vid_lbl"] = torch.randint(low=0, high=len(self.opt.categories), size=[encoded_data["code"].size(0)])
            cropped_encoded_data["vid_lbl"] = data["vid_lbl"]

        if self.opt.deblurring:
            blurred_data = blur(data, blur_sigma=self.opt.blur_sigma)
            blurred_encoded_data = self.vid_model(blurred_data, mode=f'vid_encoder')
            cropped_encoded_data["state_code"] = blurred_encoded_data["code"]
            cropped_encoded_data["inter"] = blurred_encoded_data["inter"]

        if not self.opt.rec_only:
            if self.opt.step_by_step:
                # initialize video with conditioning frames
                fake_data = {"vid": data["vid"][:, :int(crop_prop * data["vid"].size(1))]}
                # initialize codes and intermediate features from the conditioning window
                fake_encoded_data = {"code": cropped_encoded_data["code"]}
                step_encoded_data = {"inter": cropped_encoded_data["inter"]}
                if self.opt.p2p:
                    fake_encoded_data["cond_code"] = cropped_encoded_data["cond_code"]
                    fake_encoded_data["delta_length_cond"] = cropped_encoded_data["delta_length_cond"]
                    step_encoded_data["cond_inter"] = cropped_encoded_data["cond_inter"]
                for _ in tqdm(range((total_len - self.opt.cond_len - cond_len) // self.opt.z_chunk), desc="Timestep"):
                    # free a chunk if needed
                    if self.opt.p2p and fake_encoded_data["code"].size(1) > self.opt.z_len - 2 * self.opt.z_chunk:
                        fake_encoded_data["delta_length_cond"] -= (fake_encoded_data["code"].size(1) - self.opt.z_len) // self.opt.z_chunk + 2
                        fake_encoded_data["code"] = fake_encoded_data["code"][:, -(self.opt.z_len - 2 * self.opt.z_chunk):]
                    elif fake_encoded_data["code"].size(1) > self.opt.z_len - self.opt.z_chunk:
                        fake_encoded_data["code"] = fake_encoded_data["code"][:, -(self.opt.z_len - self.opt.z_chunk):]
                    # predict codes for next frame
                    fake_encoded_data = self.transformer_model(fake_encoded_data, mode='inference', total_len=fake_encoded_data["code"].size(1) + self.opt.z_chunk, show_progress=True)
                    step_encoded_data["code"] = fake_encoded_data["code"][:, -self.opt.z_chunk:]
                    # decode one frame
                    step_decoded_data = self.vid_model(step_encoded_data, mode=f'vid_step_decoder')
                    # store necessary intermediate features including new ones
                    step_encoded_data["inter"] = step_decoded_data["inter"]
                    # correct codes corresponding to new frame
                    fake_encoded_data["code"][:, -self.opt.z_chunk:] = step_decoded_data["code"]
                    # add frame to video
                    fake_data["vid"] = torch.cat([fake_data["vid"], step_decoded_data["vid"]], dim=1)
            else:
                fake_encoded_data = self.transformer_model(cropped_encoded_data, mode='inference', total_len=total_len, show_progress=True)
                t2 = time.time()
                cropped_encoded_data.update(fake_encoded_data)
                fake_data = self.vid_model(cropped_encoded_data, mode=f'vid_decoder')
                t3 = time.time()
            if self.opt.p2p:
                fake_data["vid"] = torch.cat([fake_data["vid"], data["vid"][:, -1:]], dim=1)
            if self.opt.state:
                fake_data.update(self.state_model(fake_encoded_data, mode=f'vid_decoder'))


        if not self.opt.gen_from_img:
            rec_encoded_data = {}
            rec_encoded_data["inter"] = cropped_encoded_data["inter"]
            if self.opt.p2p:
                rec_encoded_data["code"] = encoded_data["code"][:, :-self.opt.z_chunk].contiguous()
                rec_encoded_data["cond_code"] = cropped_encoded_data["cond_code"]
                rec_encoded_data["cond_inter"] = cropped_encoded_data["cond_inter"]
            else:
                rec_encoded_data["code"] = encoded_data["code"]
            if self.opt.state or self.opt.layout or self.opt.stft:
                rec_encoded_data["state_code"] = encoded_data["state_code"]
                if "interl" in encoded_data:
                    rec_encoded_data["interl"] = encoded_data["interl"]
            rec_data = self.vid_model(rec_encoded_data, mode=f'vid_decoder')
            if self.opt.p2p:
                rec_data["vid"] = torch.cat([rec_data["vid"], data["vid"][:, -1:]], dim=1)
            if self.opt.state:
                rec_data["state"] = data["state"]

        # video params
        bs = self.valid_data_info["batch_size_per_gpu"]
        fps = self.opt.fps
        normalize = True
        span = [-1, 1]
        imagenet_norm = self.opt.imagenet_norm
        dataset = self.opt.dataset

        # save videos
        cat = [self.opt.categories[i] for i in data['vid_lbl']] if self.opt.cat else None
        idx = data["vid_id"] if "vid_id" in data and self.opt.include_id else None
        real_path = os.path.join(self.opt.result_path, "real")
        save_video_batch(data["vid"], bs, global_iter, real_path, fps, normalize, imagenet_norm, span, dataset, cat=cat, idx=idx)
        if not self.opt.rec_only:
            fake_path = os.path.join(self.opt.result_path, "fake")
            save_video_batch(fake_data["vid"], bs, global_iter, fake_path, fps, normalize, imagenet_norm, span, dataset, cat=cat, idx=idx)
        if not self.opt.gen_from_img:
            rec_path = os.path.join(self.opt.result_path, "rec")
            save_video_batch(rec_data["vid"], bs, global_iter, rec_path, fps, normalize, imagenet_norm, span, dataset, cat=cat, idx=idx)
        if self.opt.deblurring:
            blur_path = os.path.join(self.opt.result_path, "blur")
            save_video_batch(blurred_data["vid"], bs, global_iter, blur_path, fps, normalize, imagenet_norm, span, dataset, cat=cat, idx=idx)

        # save state videos
        if self.opt.state:
            real_state_path = os.path.join(self.opt.result_path, "real_state")
            save_video_batch(data["vid"], bs, global_iter, real_state_path, fps, normalize, imagenet_norm, span, dataset, state=data["state"], cat=cat, idx=idx)
            if not self.opt.rec_only:
                fake_state_path = os.path.join(self.opt.result_path, "fake_state")
                save_video_batch(fake_data["vid"], bs, global_iter, fake_state_path, fps, normalize, imagenet_norm, span, dataset, state=fake_data["state"], cat=cat, idx=idx)
            if not self.opt.gen_from_img:
                rec_state_path = os.path.join(self.opt.result_path, "rec_state")
                save_video_batch(rec_data["vid"], bs, global_iter, rec_state_path, fps, normalize, imagenet_norm, span, dataset, state=rec_data["state"], cat=cat, idx=idx)

        # save layouts
        if self.opt.layout:
            real_layout_path = os.path.join(self.opt.result_path, "real_layout")
            save_video_batch(data["layout"], bs, global_iter, real_layout_path, fps, normalize, imagenet_norm, span, dataset, cat=cat, idx=idx, is_layout=True)
            fake_layout_path = os.path.join(self.opt.result_path, "fake_layout")
            save_video_batch(fake_data["layout"], bs, global_iter, fake_layout_path, fps, normalize, imagenet_norm, span, dataset, cat=cat, idx=idx, is_layout=True)

    def get_data_info(self, phase, data_type, fold=None, num_folds=None):
        from_vid = self.opt.from_vid and not self.opt.have_frames if data_type == "img" else self.opt.from_vid
        dataset =  self.engine.create_dataset(self.opt,
                                              phase=phase,
                                              fold=fold,
                                              from_vid=from_vid,
                                              load_vid=data_type == "vid")
        batch_size = self.opt.batch_size_img if data_type == "img" else self.opt.batch_size_vid
        batch_size = batch_size * self.opt.batch_size_valid_mult if phase == "valid" else batch_size
        loader_info = self.engine.create_dataloader(dataset, batch_size, self.opt.num_workers, is_train=phase == "train")
        dataloader, datasampler, batch_size_per_gpu = loader_info
        loader_iter = self.iter_fn(dataloader)
        return {"dataloader": dataloader, "datasampler": datasampler, "epoch": 0, "phase": phase, "data_type":data_type,
                "batch_size_per_gpu": batch_size_per_gpu, "loader_iter": loader_iter, "fold":fold,
                "num_folds":num_folds}

    def run(self):
        with Engine(self.opt) as engine:
            self.engine = engine
            fold_valid = self.opt.init_fold_valid if self.opt.num_folds_valid is not None else None
            data_type = "img" if self.opt.gen_from_img else "vid"
            self.valid_data_info = self.get_data_info("valid", data_type, fold=fold_valid, num_folds=self.opt.num_folds_valid)

            is_main = self.engine.is_main
            
            self.vid_model_on_one_gpu = QVidModel(self.qvid_opt, is_train=False, is_main=is_main, logger=None)
            self.vid_model = engine.data_parallel(self.vid_model_on_one_gpu)
            self.vid_model.eval()

            if not self.opt.rec_only:
                self.transformer_model_on_one_gpu = Transformer(self.opt, is_train=False, is_main=is_main, logger=None)
                self.transformer_model = engine.data_parallel(self.transformer_model_on_one_gpu)
                self.transformer_model.eval()

            if self.opt.state:
                self.state_model_on_one_gpu = StateModel(self.state_opt, is_train=False, is_main=is_main, logger=None)
                self.state_model = engine.data_parallel(self.state_model_on_one_gpu)
                self.state_model.eval()

            if self.opt.stft:
                self.stft_model_on_one_gpu = StftModel(self.stft_ae_opt, is_train=False, is_main=is_main, logger=None)
                self.stft_model = engine.data_parallel(self.stft_model_on_one_gpu)
                self.stft_model.eval()

            for global_iter in tqdm(range(self.opt.n_iter), desc="Batch"):
                data = self.next_batch(self.valid_data_info)
                if self.opt.gen_from_img:
                    data["vid"] = data.pop("img").unsqueeze(1)
                self.generate_vid(data, global_iter)

            print('Generation was successfully finished.')


def save_video_batch(vid, bs, global_iter, path, fps, normalize, imagenet_norm, span, dataset, state=None, cat=None, idx=None, is_layout=False):
    # postprocess
    vid = vid.cpu()
    if is_layout:
        shape = vid.shape[:2]
        vid = vid.view(-1, *vid.shape[2:])
        if vid.ndim == 4:
            seg = vid.max(1, keepdim=True)[1]
        else:
            seg = vid.unsqueeze(1)
        colormap = np.array([[128., 64., 128.], [244.,  35., 232.], [230., 150., 140.],[ 70.,  70.,  70.],[102., 102., 156.],[153., 153., 153.],[250., 170.,  30.],[220., 220.,   0.],[107., 142., 135.],[152., 251., 152.],[230., 150., 140.],[220.,  20.,  60.],[255.,   0.,   0.],[  0.,   0., 142.],[  0.,   0.,  70.],[  0.,  60., 100.],[  0.,  80., 100.],[  0.,   0., 230.],[119.,  11.,  32.]])
        colormap = colormap / 255.
        vid = color_transfer(seg, colormap)
        vid = vid.view(*shape, *vid.shape[1:])
        vid = 0.5 + vid / 2
    else:
        if normalize and imagenet_norm:
            # tensor should be in [-1 1]
            vid *= torch.tensor([[[[0.229]], [[0.224]], [[0.225]]]])
            vid += torch.tensor([[[[0.485]], [[0.456]], [[0.406]]]])
            vid = vid.clamp(0, 1)
        elif normalize:
            vid = vid.clamp(span[0], span[1])
            vid = (vid - span[0]) / (span[1] - span[0])
    vid = (vid.permute(0, 1, 3, 4, 2) * 255).to(dtype=torch.uint8)

    # include state
    if state is not None:
        if dataset == "bair":
            for i in range(vid.size(0)):
                for j in range(vid.size(1)):
                    x, y = state[i, j]
                    x, y = min(int(64 * x), 63), min(int(64 * y), 63)
                    vid[i, j] = draw_cross(vid[i, j], x ,y)
        if dataset == "bairhd":
            for i in range(vid.size(0)):
                for j in range(vid.size(1)):
                    x, y = state[i, j]
                    x, y = min(int(256 * x), 255), min(int(256 * y), 255)
                    vid[i, j] = draw_cross(vid[i, j], x ,y)

    # save
    mkdir(path)
    for i in range(vid.size(0)):
        suffix = '' if cat is None else f'_{cat[i]}'
        suffix += '' if idx is None else f'_{idx[i]}'
        vid_id = bs * global_iter + i
        filename = os.path.join(path, f"vid_{vid_id:05d}{suffix}.mp4")
        torchvision.io.write_video(filename, vid[i], fps)


def draw_cross(img, x, y):
    height, width = img.shape[:2]
    left = x - 1
    right = x + 1
    top = y - 1
    bot = y + 1
    img[y, x] = img[y, x] * 0 + 255
    if left >= 0:
        img[y, left] = img[y, left] * 0 + 255
        if bot < height:
            img[bot, left] = img[bot, left] * 0
        if top >= 0:
            img[top, left] = img[top, left] * 0
    if right < width:
        img[y, right] = img[y, right] * 0 + 255
        if bot < height:
            img[bot, right] = img[bot, right] * 0
        if top >= 0:
            img[top, right] = img[top, right] * 0
    if top >= 0:
        img[top, x] = img[top, x] * 0 + 255
    if bot < height:
        img[bot, x] = img[bot, x] * 0 + 255
    return img


def square_trajectory(init_state, vid_len):
    state = init_state.repeat(1, vid_len, 1)
    is_inside = lambda u, v: 0.2 <= u and 0.2 <= v and u < 0.8 and v < 0.8
    for i in range(state.size(0)):
        x, y = state[i, 0].clone()
        step = 10 / 64
        delta = [(0, -step), (step, 0), (0, step), (-step, 0)]
        t = 0
        dx, dy = delta[t]
        for j in range(1, vid_len):
            while not is_inside(x + dx, y + dy):
                t = (t + 1) % 4
                dx, dy = delta[t]
            x += dx
            y += dy
            state[i, j, 0] = x
            state[i, j, 1] = y
    return {"state": state}

def blur(data, blur_sigma=10):
    vid = data["vid"]
    bs, t = vid.shape[:2]
    img = data["vid"].view(-1, *vid.shape[2:])
    s = blur_sigma
    k = int(3 * s) + 1 if int(3 * s) % 2 == 0 else int(3 * s)
    blur_t = transforms.GaussianBlur(kernel_size=max(3, min(k, 13)), sigma=s)
    blurred_img = blur_t(img)
    blurred_vid = blurred_img.view(bs, t, *vid.shape[2:])
    return {"vid": blurred_vid}

if __name__ == "__main__":
    opt = Options().parse(load_qvid_generator=True, load_transformer=True, load_state_estimator=True, load_stft_ae=True, save=False)
    Generator(opt).run()
