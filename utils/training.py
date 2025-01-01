import random
import os
import numpy as np
import torch
from tqdm import tqdm
import time

from dataset.kitti_dataset import KITTI
from dataset.get_data_loader import get_trainval_data_loader_fn
from model.pointpillar import PointPillar
from model.loss_fn import Loss3DDetection
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, cfg):
        cfg.mode = "training"
        self.seed = cfg.seed
        self.max_epoch = cfg.max_epoch
        self.log_freq = cfg.log_freq
        self.ckpt_freq_epoch = cfg.ckpt_freq_epoch

        self.setup_seed()
        train_dataset = KITTI(cfg, split="train")
        val_dataset = KITTI(cfg, split="val")
        self.train_dataloader = get_trainval_data_loader_fn(dataset=train_dataset,
                                                            batch_size=cfg.batch_size)
        self.val_dataloader = get_trainval_data_loader_fn(val_dataset,
                                                          cfg.batch_size, shuffle=False)

        self.model = PointPillar(cfg)

        self.use_cuda = cfg.device == "gpu" and torch.cuda.is_available()
        if self.use_cuda:
            self.model = self.model.cuda()
        self.loss_func = Loss3DDetection(cfg)

        init_lr = cfg.init_lr
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                           lr=init_lr,
                                           betas=(0.95, 0.99),
                                           weight_decay=0.01)
        max_iters = len(self.train_dataloader) * cfg.max_epoch
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             max_lr=init_lr * 10,
                                                             total_steps=max_iters,
                                                             pct_start=0.4,
                                                             anneal_strategy="cos",
                                                             cycle_momentum=True,
                                                             base_momentum=0.95 * 0.895,
                                                             max_momentum=0.95,
                                                             div_factor=10)
        saved_logs_path = os.path.join(cfg.root_path, "model", "point_pillar", "summary")
        os.makedirs(saved_logs_path, exist_ok=True)
        self.writer = SummaryWriter(saved_logs_path)
        self.saved_model_path = os.path.join(cfg.root_path, "model", "point_pillar", "checkpoints")
        os.makedirs(self.saved_model_path, exist_ok=True)

    def setup_seed(self, deterministic=True):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def save_summary(self, loss, global_step, tag, lr=None, momentum=None):
        self.writer.add_scalar(f"{tag}/'loss'", loss, global_step)
        if lr is not None:
            self.writer.add_scalar("lr", lr, global_step)
        if momentum is not None:
            self.writer.add_scalar("momentum", momentum, global_step)
        print(tag + " Loss: ", loss.detach().cpu().numpy())

    def load_data_gpu(self, data_dict):
        if self.use_cuda:
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].cuda()

    def save_model(self, epoch):
        param_dict = {"model": self.model.state_dict(),
                      "optimizer_state_dict": self.optimizer.state_dict(),
                      "epoch": epoch}
        torch.save(param_dict, os.path.join(self.saved_model_path, str((epoch + 1)) + "model.pth"))

    def load_model(self):
        recent_epoch_num = 0
        model_file_names = [int(file_name[:-9]) for file_name in
                            os.listdir(self.saved_model_path)
                            if os.path.isfile(os.path.join(self.saved_model_path, file_name) and len(file_name) >9)]
        if len(model_file_names) > 0:
            recent_epoch_num = max(model_file_names)
        model_path = os.path.join(self.saved_model_path, str(recent_epoch_num) + "model.pth")
        recent_train_epoch = 0
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            try:
                key = "model"
                self.model.load_state_dict(checkpoint[key])
                key = "optimizer_state_dict"
                self.optimizer.load_state_dict(checkpoint[key])
                recent_train_epoch = checkpoint["epoch"]
                print(f"The pretrained model is loaded from {model_path}")
            except:
                print("The pretrained model in the model path is not matching. The model is started from scratch.")
        else:
            print("The model is started from scratch.")
        return recent_train_epoch

    def train(self):
        start_epoch = self.load_model()
        start_time = time.time()
        for epoch in range(start_epoch, self.max_epoch):
            print("|" + "-" * 25, epoch, "-" * 25 + "|")
            train_step, val_step = 0, 0
            self.model.train()
            for i, data_dict in enumerate(tqdm(self.train_dataloader)):
                self.load_data_gpu(data_dict)

                self.optimizer.zero_grad()
                predict_res_batch = self.model(data_dict)

                loss = self.loss_func(predict_res_batch)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                global_step = epoch * len(self.train_dataloader) + train_step + 1

                if global_step % self.log_freq == 0:
                    self.save_summary(loss, global_step, "train",
                                      lr=self.optimizer.param_groups[0]["lr"],
                                      momentum=self.optimizer.param_groups[0]["betas"][0])
                train_step += 1
            if (epoch + 1) % self.ckpt_freq_epoch == 0:
                self.save_model(epoch)

            if (epoch + 1) % 5 == 0:
                # perform validation test
                self.model.eval()
                with torch.no_grad():
                    for i, data_dict in enumerate(tqdm(self.val_dataloader)):
                        self.load_data_gpu(data_dict)

                        predict_val_res_batch = self.model(data_dict)

                        loss = self.loss_func(predict_val_res_batch)

                        global_step = epoch * len(self.val_dataloader) + val_step + 1
                        if global_step % self.log_freq == 0:
                            self.save_summary(loss, global_step, "val")
                        val_step += 1
        end_time = time.time()
        print("It takes {} hours to train the model for {} epochs.".format((end_time - start_time) / 3600.0,
                                                                           self.max_epoch - start_epoch))
