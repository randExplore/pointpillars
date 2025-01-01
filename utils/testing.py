import os
import numpy as np
import torch

from dataset.kitti_dataset import KITTI
from model.pointpillar import PointPillar
from utils.visualization import plot_all


class InferDataset:
    def __init__(self, cfg):
        cfg.data_augmentaion = False
        cfg.split = "val"
        cfg.mode = "testing"
        self.val_dataset = KITTI(cfg, split="val")

        self.model = PointPillar(cfg)
        self.model.eval()
        self.use_cuda = cfg.device == "gpu" and torch.cuda.is_available()
        if self.use_cuda:
            self.model = self.model.cuda()
        self.saved_model_path = os.path.join(cfg.root_path, "model", "point_pillar", "checkpoints")
        self.load_model()

    def load_model(self):
        model_path = os.path.join(self.saved_model_path, "160model.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            try:
                key = "model"
                self.model.load_state_dict(checkpoint[key])
                print(f"The pretrained model is loaded from {model_path}")
            except:
                print("The pretrained model in the model path is not matching. The model is started from scratch.")
        else:
            print("The model is started from scratch.")

    def load_data_gpu(self, data_dict):
        if self.use_cuda:
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].cuda()

    def convert_data_dict(self, data_dict):
        model_input_dict = dict(
            batched_pts=[torch.from_numpy(data_dict["pts"])],
            batched_gt_bboxes=[torch.from_numpy(data_dict["gt_bboxes_3d"])],
            batched_labels=[torch.from_numpy(data_dict["gt_labels"])],
            batched_names=[data_dict["gt_object_name"]],
            batched_difficulty=[data_dict["difficulty"]],
            batched_img_info=[data_dict["image_info"]],
            batched_calib_info=[data_dict["calib_info"]]
        )  # follows the same format in the data loader
        self.load_data_gpu(model_input_dict)
        return model_input_dict

    def __call__(self, dataset_idx=None):
        if dataset_idx is None:
            dataset_idx = np.random.choice(len(self.val_dataset), 1).squeeze()
        assert 0 <= dataset_idx <= len(self.val_dataset)
        cur_data_dict = self.val_dataset[dataset_idx]
        model_input_dict = self.convert_data_dict(cur_data_dict)
        print("|------------Start inferring the data with idx {}-----------|".format(dataset_idx))
        predict_res_dict = self.model(model_input_dict)[0]
        plot_all(predict_res_dict, cur_data_dict, save_pred_result=True)
