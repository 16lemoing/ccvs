from itertools import cycle
import random

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

class FrameAutoencoderTrainer:
    def __init__(self, opt):
        self.opt = opt["qvid_generator"]
        self.extra_dataset_opt = opt["extra_base"]
        self.iter_fn = cycle if self.opt.iter_function == "cycle" else iter

    def next_batch(self, data_info):
        try:
            return next(data_info["loader_iter"])
        except StopIteration:
            if data_info["num_folds"] is not None:
                num_folds = data_info["num_folds"]
                epoch = data_info["epoch"] + 1 / num_folds
                fold = (data_info["fold"] + 1) % num_folds
                phase, data_type = data_info["phase"], data_info["data_type"]
                # free memory from previous fold before loading the next
                for k in data_info:
                    data_info[k] = None
                new_data_info = self.get_data_info(phase, data_type, fold, num_folds)
                new_data_info["epoch"] = epoch
                for k in data_info:
                    data_info[k] = new_data_info[k]
            else:
                data_info["epoch"] += 1
                if self.engine.distributed:
                    data_info["datasampler"].set_epoch(data_info["epoch"])
                data_info["loader_iter"] = self.iter_fn(data_info["dataloader"])
            return next(data_info["loader_iter"])

    def reinit_batches(self, data_info):
        data_info["loader_iter"] = self.iter_fn(data_info["dataloader"])

    def step_model(self, data, data_type, global_iter, log):
        # generator step
        self.opt_g.zero_grad()
        loss_gen, fake_data = self.vid_model(data, mode=f'{data_type}_to_{data_type}_generator', log=log, global_iter=global_iter)
        loss_gen = self.engine.all_reduce_tensor(loss_gen)
        loss_gen.backward()
        self.opt_g.step()

        # discriminator step
        self.opt_d.zero_grad()
        loss_dis = self.vid_model(data, fake_data=fake_data, mode=f'{data_type}_discriminator', log=log, global_iter=global_iter)
        loss_dis = self.engine.all_reduce_tensor(loss_dis)
        loss_dis.backward()
        self.opt_d.step()

        # generator regularization
        if data_type == "img" and self.opt.g_reg_every is not None and global_iter % self.opt.g_reg_every == 0:
            self.opt_g.zero_grad()
            loss_reg = self.vid_model(data, mode=f'img_to_img_generator_reg', log=log, global_iter=global_iter)
            loss_reg.backward()
            self.opt_g.step()

        # discriminator regularization
        if self.opt.d_reg_every is not None and global_iter % self.opt.d_reg_every == 0:
            self.opt_d.zero_grad()
            loss_reg = self.vid_model(data, fake_data=fake_data, mode=f'{data_type}_discriminator_reg', log=log, global_iter=global_iter)
            loss_reg.backward()
            self.opt_d.step()

        # exponential moving average
        self.vid_model_on_one_gpu.accumulate()


    def get_data_info(self, phase, data_type, fold=None, num_folds=None):
        from_vid = self.opt.from_vid and not self.opt.have_frames if data_type == "img" else self.opt.from_vid
        opt = self.extra_dataset_opt if data_type == "vid" and self.opt.use_extra_dataset else self.opt
        dataset =  self.engine.create_dataset(opt,
                                              phase=phase,
                                              fold=fold,
                                              from_vid=from_vid,
                                              load_vid=data_type == "vid")
        if data_type == "img":
            if phase == "train":
                batch_size = self.opt.batch_size_img // (self.opt.n_consecutive_img + (1 if self.opt.load_elastic_view else 0))
            else:
                batch_size = self.opt.batch_size_img
        else:
            batch_size = self.opt.batch_size_vid
        loader_info = self.engine.create_dataloader(dataset, batch_size, self.opt.num_workers, is_train=phase == "train")
        dataloader, datasampler, batch_size_per_gpu = loader_info
        loader_iter = self.iter_fn(dataloader)
        return {"dataloader": dataloader, "datasampler": datasampler, "epoch": 0, "phase": phase, "data_type":data_type,
                "batch_size_per_gpu": batch_size_per_gpu, "loader_iter": loader_iter, "fold":fold,
                "num_folds":num_folds}

    def run(self):
        with Engine(self.opt) as engine:
            self.engine = engine
            fold_train = self.opt.init_fold_train if self.opt.num_folds_train is not None else None
            fold_train = random.randrange(self.opt.num_folds_train) if (self.opt.num_folds_train and self.opt.random_fold_train) else fold_train
            self.train_img_data_info = self.get_data_info("train", "img", fold=fold_train, num_folds=self.opt.num_folds_train)
            if self.opt.is_seq:
                self.train_vid_data_info = self.get_data_info("train", "vid", fold=fold_train, num_folds=self.opt.num_folds_train)

            is_main = self.engine.is_main
            logger = Logger(self.opt) if is_main else None
            
            self.vid_model_on_one_gpu = QVidModel(self.opt, is_train=True, is_main=is_main, logger=logger)
            self.opt_g = self.vid_model_on_one_gpu.opt_g
            self.opt_d = self.vid_model_on_one_gpu.opt_d

            self.vid_model = engine.data_parallel(self.vid_model_on_one_gpu)
            self.vid_model.train()

            start_iter = int(self.opt.which_iter) + 1 if self.opt.cont_train else 0
            for global_iter in range(start_iter, self.opt.n_iter):

                # train
                log = global_iter % self.opt.log_freq == 0 and is_main
                img_data = self.next_batch(self.train_img_data_info)
                self.step_model(img_data, "img", global_iter, log)
                if self.opt.is_seq and global_iter % self.opt.vid_step_every == 0:
                    vid_data = self.next_batch(self.train_vid_data_info)
                    self.step_model(vid_data, "vid", global_iter, log)
                if log:
                    log_string = f"Img epoch {self.train_img_data_info['epoch']:07.2f}"
                    log_string += f" fold {self.train_img_data_info['fold']}" if self.train_img_data_info['fold'] is not None else ""
                    log_string += f", Vid epoch {self.train_vid_data_info['epoch']:07.2f}" if self.opt.is_seq else ""
                    log_string += f" fold {self.train_vid_data_info['fold']}" if self.opt.is_seq and  self.train_vid_data_info['fold'] is not None else ""
                    log_string += f", Iteration {global_iter:05d}/{self.opt.n_iter:05d}"

                    print(log_string)

                # checkpoint
                if self.opt.save_freq > 0 and global_iter % self.opt.save_freq == 0 and is_main:
                    self.vid_model_on_one_gpu.save_model(global_iter, latest=False)
                if self.opt.save_latest_freq > 0 and global_iter % self.opt.save_latest_freq == 0 and is_main:
                    self.vid_model_on_one_gpu.save_model(global_iter, latest=True)

            if is_main:
                self.vid_model_on_one_gpu.save_model(self.opt.n_iter - 1, latest=True)

            print('Training was successfully finished.')


if __name__ == "__main__":
    opt = Options().parse(load_qvid_generator=True, load_extra_base=True, save=True)
    FrameAutoencoderTrainer(opt).run()
