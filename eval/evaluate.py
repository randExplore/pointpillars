import numpy as np
import os
import torch
from tqdm import tqdm
from utils.visualization import filter_bbox_in_image_range, filter_bbox_in_lidar_range
from dataset.kitti_dataset import KITTI
from dataset.get_data_loader import get_trainval_data_loader_fn
from model.pointpillar import PointPillar
import time
from .do_eval import get_official_eval_result


class Evaluation:
    def __init__(self, cfg):
        cfg.mode = "testing"
        self.pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)
        self.object_class = cfg.class_name
        self.label2class = {v: k for k, v in self.object_class.items()}
        self.gt_annos = dict()
        self.pred_annos = dict()

        self.val_dataset = KITTI(cfg, split="val")
        self.val_dataloader = get_trainval_data_loader_fn(dataset=self.val_dataset,
                                        batch_size=1,
                                        num_workers=0,
                                        shuffle=False)

        self.model = PointPillar(cfg)
        self.use_cuda = cfg.device == "gpu" and torch.cuda.is_available()
        if self.use_cuda:
            self.model = self.model.cuda()
        self.saved_model_path = os.path.join(cfg.root_path, "model", "point_pillar", "checkpoints")

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

    def process_gt_label(self):
        data_info = self.val_dataset.data_infos
        for file_id in data_info:
            cur_info_dict = data_info[file_id]
            self.gt_annos[file_id] = cur_info_dict

    def process_batch_pred_res(self, data_dict, batch_results):
        for j, result in enumerate(batch_results):
            format_result = {
                "name": [],
                "truncated": [],
                "occluded": [],
                "alpha": [],
                "bbox": [],
                "dimensions": [],
                "location": [],
                "rotation_y": [],
                "score": []
            }

            calib_info = data_dict["batched_calib_info"][j]
            tr_velo_to_cam = calib_info["Tr_velo_to_cam"].astype(np.float32)
            r0_rect = calib_info["R0_rect"].astype(np.float32)
            P2 = calib_info["P2"].astype(np.float32)
            image_shape = data_dict["batched_img_info"][j]["image_shape"]
            idx = data_dict["batched_img_info"][j]["image_idx"]
            result_filter = filter_bbox_in_image_range(result, tr_velo_to_cam, r0_rect, P2, image_shape)
            result_filter = filter_bbox_in_lidar_range(result_filter, self.pcd_limit_range)

            lidar_bboxes = result_filter["bboxes3D"]
            labels, scores = result_filter["labels"], result_filter["scores"]
            bboxes2d, camera_bboxes = result_filter["bboxes2d"], result_filter["camera_bboxes"]
            for lidar_bbox, label, score, bbox2d, camera_bbox in \
                    zip(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes):
                format_result["name"].append(self.label2class[label])
                format_result["truncated"].append(0.0)
                format_result["occluded"].append(0)
                alpha = camera_bbox[6] - np.arctan2(camera_bbox[0], camera_bbox[2])
                format_result["alpha"].append(alpha)
                format_result["bbox"].append(bbox2d)
                format_result["dimensions"].append(camera_bbox[3:6])
                format_result["location"].append(camera_bbox[:3])
                format_result["rotation_y"].append(camera_bbox[6])
                format_result["score"].append(score)

            self.pred_annos[idx] = {k: np.array(v) for k, v in format_result.items()}

    def cumulate_pred_labels(self):
        self.load_model()
        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(self.val_dataloader)):
                self.load_data_gpu(data_dict)
                batch_results = self.model(data_dict)
                self.process_batch_pred_res(data_dict, batch_results)
        end_time = time.time()
        print("It takes {} mins to get the predicted "
              "bboxes from the model for {} number of data.".format((end_time - start_time) / 60.0,
                                                                           len(self.val_dataset)))

    def evaluate(self):
        self.process_gt_label()
        self.cumulate_pred_labels()
        gt_annos = []
        dt_annos = []
        for file_name in sorted(self.pred_annos):
            gt_annos.append(self.gt_annos[file_name]["annotation"])
            dt_annos.append(self.pred_annos[file_name])

        current_class = ["Pedestrian", "Cyclist", "Car"]
        print("|------------Start evaluating-----------|")
        result, ret_dict = get_official_eval_result(gt_annos, dt_annos, current_class)
        print("Evaluation results:\n", result)
