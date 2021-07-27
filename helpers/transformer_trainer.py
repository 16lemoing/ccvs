import torch
import torchvision.transforms as transforms
import math
import random
from itertools import cycle

import warnings
warnings.filterwarnings("ignore")

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
code_dir = os.path.dirname(current_dir)
sys.path.insert(0, code_dir)

from tools.options import Options
from tools.engine import Engine
from tools.logger import Logger
from models.skip_vid_generator.models.quantized_video_model import QVidModel
from models.skip_vid_generator.models.state_model import StateModel
from models.skip_vid_generator.models.transformer_model import Transformer
from models.skip_vid_generator.models.stft_model import StftModel

class TransformerTrainer:
    def __init__(self, opt):
        self.opt = opt["transformer"]
        self.qvid_opt = opt["qvid_generator"]
        self.state_opt = opt["state_estimator"]
        self.extra_dataset_opt = opt["extra_base"]
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


    def step_model(self, data, data_type, global_iter, log):
        encoded_data = self.vid_model(data, mode=f'{data_type}_encoder', log=log, suffix="_train", global_iter=global_iter)

        if self.opt.state:
            data_for_state = data if self.state_opt.quantize_only else encoded_data
            state_encoded_data = self.state_model(data_for_state, mode=f'{data_type}_encoder', log=log, global_iter=global_iter)
            encoded_data.update(state_encoded_data)

        if self.opt.stft:
            stft_encoded_data = self.stft_model(data, mode=f'{data_type}_encoder', log=log, global_iter=global_iter)
            encoded_data.update(stft_encoded_data)

        if self.opt.p2p and data_type == "vid":
            encoded_data["cond_code"] = encoded_data["code"][:, -self.opt.z_chunk:]
            encoded_data["code"] = encoded_data["code"][:, :-self.opt.z_chunk]
            encoded_data["delta_length_cond"] = data["delta_length"]

        if self.opt.cat:
            if "vid_lbl" not in data:
                data["vid_lbl"] = torch.randint(low=0, high=len(self.opt.categories), size=[data[data_type].size(0)])
            encoded_data["vid_lbl"] = data["vid_lbl"]

        if self.opt.deblurring:
            blurred_data = blur(data, blur_sigma=self.opt.blur_sigma)
            blurred_encoded_data = self.vid_model(blurred_data, mode=f'{data_type}_encoder', log=log, suffix="_train", global_iter=global_iter)
            encoded_data["state_code"] = blurred_encoded_data["code"]

        self.opt_t.zero_grad()
        t_loss = self.transformer_model(encoded_data, mode=f'transformer', prefix=f"{data_type}_", log=log, global_iter=global_iter)
        t_loss = self.engine.all_reduce_tensor(t_loss)
        t_loss.backward()
        self.opt_t.step()


    def get_data_info(self, phase, data_type, fold=None, num_folds=None):
        from_vid = self.opt.from_vid and not self.opt.have_frames if data_type == "img" else self.opt.from_vid
        opt = self.extra_dataset_opt if data_type == "img" and self.opt.use_extra_dataset else self.opt
        dataset =  self.engine.create_dataset(opt,
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


    def update_lr(self, global_iter):
        if self.opt.lr_decay:
            if global_iter < self.opt.warmup_iter:
                # linear warmup
                lr_mult = float(global_iter) / float(max(1, self.opt.warmup_iter))
            else:
                # cosine learning rate decay
                progress = float(global_iter - self.opt.warmup_iter) / float(max(1, self.opt.n_iter - self.opt.warmup_iter))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = self.opt.lr * lr_mult
            for param_group in self.opt_t.param_groups:
                param_group['lr'] = lr


    def run(self):
        with Engine(self.opt) as engine:
            self.engine = engine
            fold_train = self.opt.init_fold_train if self.opt.num_folds_train is not None else None
            fold_train = random.randrange(self.opt.num_folds_train) if (self.opt.num_folds_train and self.opt.random_fold_train) else fold_train
            data_type = "vid" if self.opt.is_seq else "img"
            self.train_data_info = self.get_data_info("train", data_type, fold=fold_train, num_folds=self.opt.num_folds_train)
            if self.opt.use_extra_dataset:
                self.train_extra_data_info = self.get_data_info("train", "img")

            is_main = self.engine.is_main
            logger = Logger(self.opt) if is_main else None
            
            self.vid_model_on_one_gpu = QVidModel(self.qvid_opt, is_train=False, is_main=is_main, logger=logger)
            self.state_model_on_one_gpu = StateModel(self.state_opt, is_train=False, is_main=is_main, logger=logger) if self.opt.state else None
            self.stft_model_on_one_gpu = StftModel(self.stft_ae_opt, is_train=True, is_main=is_main, logger=logger) if self.opt.stft else None
            self.transformer_model_on_one_gpu = Transformer(self.opt, is_train=True, is_main=is_main, logger=logger)
            self.opt_t = self.transformer_model_on_one_gpu.opt_t

            self.vid_model = engine.data_parallel(self.vid_model_on_one_gpu)
            self.state_model = engine.data_parallel(self.state_model_on_one_gpu) if self.opt.state else None
            self.stft_model = engine.data_parallel(self.stft_model_on_one_gpu) if self.opt.stft else None
            self.transformer_model = engine.data_parallel(self.transformer_model_on_one_gpu)

            self.vid_model.eval()
            self.state_model.eval() if self.opt.state else None
            self.stft_model.eval() if self.opt.stft else None
            self.transformer_model.train()

            start_iter = int(self.opt.which_iter) + 1 if self.opt.cont_train else 0
            for global_iter in range(start_iter, self.opt.n_iter):

                # train
                log = self.opt.log_freq is not None and global_iter % self.opt.log_freq == 0 and is_main
                data = self.next_batch(self.train_data_info)
                self.step_model(data, data_type, global_iter, log)
                if self.opt.use_extra_dataset:
                    data = self.next_batch(self.train_extra_data_info)
                    self.step_model(data, "img", global_iter, False)
                if log:
                    print(f"Epoch {self.train_data_info['epoch']:05d}, Iteration {global_iter:05d}/{self.opt.n_iter:05d}")

                # checkpoint
                if self.opt.save_freq > 0 and global_iter % self.opt.save_freq == 0 and is_main:
                    self.transformer_model_on_one_gpu.save_model(global_iter, latest=False)
                if self.opt.save_latest_freq > 0 and global_iter % self.opt.save_latest_freq == 0 and is_main:
                    self.transformer_model_on_one_gpu.save_model(global_iter, latest=True)

                # learning rate
                self.update_lr(global_iter + 1)

            if is_main:
                self.transformer_model_on_one_gpu.save_model(self.opt.n_iter - 1, latest=True)

            print('Training was successfully finished.')


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
    opt = Options().parse(load_qvid_generator=True, load_transformer=True, load_extra_base=True, load_state_estimator=True, load_stft_ae=True, save=True)
    TransformerTrainer(opt).run()