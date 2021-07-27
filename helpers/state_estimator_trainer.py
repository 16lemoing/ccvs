import torch
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
from models.skip_vid_generator.models.state_model import StateModel

class StateEstimatorTrainer:
    def __init__(self, opt):
        self.qvid_opt = opt["qvid_generator"]
        self.opt = opt["state_estimator"]
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

    def step_model(self, data, global_iter, log):
        # encode data
        if not self.opt.quantize_only:
            encoded_data = self.vid_model(data, mode='img_encoder', log=log, global_iter=global_iter)
            data.update(encoded_data)

        # state estimator step
        self.opt_s.zero_grad()
        loss = self.state_model(data, mode='state_estimator', log=log, global_iter=global_iter)
        loss = self.engine.all_reduce_tensor(loss)
        loss.backward()
        self.opt_s.step()

        return loss.detach().cpu()

    @torch.no_grad()
    def eval_model_img(self, data, global_iter, log):
        # encode data
        if not self.opt.quantize_only:
            encoded_data = self.vid_model(data, mode='img_encoder', log=log, global_iter=global_iter)
            data.update(encoded_data)

        # eval step
        loss_rec = self.state_model(data, mode='eval_state_estimator', log=log, global_iter=global_iter)
        loss_rec = self.engine.all_reduce_tensor(loss_rec)

        return loss_rec

    def get_data_info(self, phase, data_type, fold=None, num_folds=None):
        from_vid = self.opt.from_vid and not self.opt.have_frames if data_type == "img" else self.opt.from_vid
        opt = self.extra_dataset_opt if data_type == "vid" and self.opt.use_extra_dataset else self.opt
        dataset =  self.engine.create_dataset(opt,
                                              phase=phase,
                                              fold=fold,
                                              from_vid=from_vid,
                                              load_vid=data_type == "vid")
        if data_type == "img":
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
            self.valid_img_data_info = self.get_data_info("valid", "img") if self.opt.n_iter_eval is not None else None

            is_main = self.engine.is_main
            logger = Logger(self.opt) if is_main else None

            self.state_model_on_one_gpu = StateModel(self.opt, is_train=True, is_main=is_main, logger=logger)
            self.opt_s = self.state_model_on_one_gpu.opt_s
            self.state_model = engine.data_parallel(self.state_model_on_one_gpu)
            self.state_model.train()

            if not self.opt.quantize_only:
                self.vid_model_on_one_gpu = QVidModel(self.qvid_opt, is_train=False, is_main=is_main, logger=logger)
                self.vid_model = engine.data_parallel(self.vid_model_on_one_gpu)
                self.vid_model.eval()

            best_eval = None
            rec_train = []

            start_iter = int(self.opt.which_iter) + 1 if self.opt.cont_train else 0
            for global_iter in range(start_iter, self.opt.n_iter):

                # eval
                if self.opt.n_iter_eval is not None and global_iter % self.opt.n_iter_eval == 0:
                    self.state_model.eval()
                    rec_eval = []
                    self.reinit_batches(self.valid_img_data_info)
                    for i, img_data in enumerate(self.valid_img_data_info["loader_iter"]):
                        rec_eval.append(self.eval_model_img(img_data, global_iter, i == 0))
                        if self.opt.max_eval_batches is not None and i > self.opt.max_eval_batches:
                            break
                    rec_eval = torch.mean(torch.stack(rec_eval))
                    self.reinit_batches(self.valid_img_data_info)
                    self.state_model.train()
                    if is_main:
                        self.state_model_on_one_gpu.logger.log_scalar(f"state_estimator/eval_rec_loss", rec_eval, global_iter)
                        print(f"[Eval] Rec {rec_eval:.5f}, Iteration {global_iter:05d}/{self.opt.n_iter:05d}")
                        is_better = best_eval is None or rec_eval < best_eval
                        best_eval = rec_eval if is_better else best_eval
                        if is_better:
                            self.state_model_on_one_gpu.save_model(global_iter, best=True)

                # train
                log = global_iter % self.opt.log_freq == 0 and is_main
                img_data = self.next_batch(self.train_img_data_info)
                rec_train.append(self.step_model(img_data, global_iter, log))

                if log:
                    log_string = f"[Train] Rec {torch.mean(torch.stack(rec_train)):.5f} "
                    log_string += f"Epoch {self.train_img_data_info['epoch']:07.2f}"
                    log_string += f" fold {self.train_img_data_info['fold']}" if self.train_img_data_info['fold'] is not None else ""
                    log_string += f", Iteration {global_iter:05d}/{self.opt.n_iter:05d}"
                    print(log_string)
                    rec_train = []

                # checkpoint
                if self.opt.save_freq > 0 and global_iter % self.opt.save_freq == 0 and is_main:
                    self.state_model_on_one_gpu.save_model(global_iter, latest=False)
                if self.opt.save_latest_freq > 0 and global_iter % self.opt.save_latest_freq == 0 and is_main:
                    self.state_model_on_one_gpu.save_model(global_iter, latest=True)

            if is_main:
                self.state_model_on_one_gpu.save_model(self.opt.n_iter - 1, latest=True)

            print('Training was successfully finished.')


if __name__ == "__main__":
    opt = Options().parse(load_qvid_generator=True, load_state_estimator=True, save=True)
    StateEstimatorTrainer(opt).run()
