import argparse
import os
import numpy as np
import torch
from dataset.get_data_info import PreprocessKittiDataset
from dataset.kitti_dataset import KITTI
from utils.preprocess_fns import bbox_camera2lidar
from model.pointpillar import PointPillar
from utils.visualization import plot_all


class Inference:
    def __init__(self, cfg, point_cloud_file_path, image_file_path,
                 calibration_file_path, label_info_file_path=None):
        cfg.mode = "testing"
        self.object_category = cfg.class_name
        self.pc_range = cfg.pc_range
        self.point_cloud_file_path = point_cloud_file_path
        self.image_file_path = image_file_path
        self.calibration_file_path = calibration_file_path
        self.ground_truth_file_path = label_info_file_path
        self.data_dict = self._prepare_data()

        self.model = PointPillar(cfg)
        self.model.eval()
        self.use_cuda = cfg.device == "gpu" and torch.cuda.is_available()
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model_input_dict = self._convert_data_dict(self.data_dict)
        self.saved_model_path = os.path.join(cfg.root_path, "model", "point_pillar", "checkpoints")
        self._load_model()

    def __call__(self):
        print("|------------Start the inference----------------|")
        predict_res_dict = self.model(self.model_input_dict)[0]
        plot_all(predict_res_dict, self.data_dict, save_pred_result=True)

    def _prepare_data(self):
        file_id = os.path.splitext(os.path.basename(self.image_file_path))[0]
        image_info_dict = PreprocessKittiDataset.get_img_info(file_id, self.image_file_path)

        calib_dict = PreprocessKittiDataset.get_calib_data_dict(self.calibration_file_path)

        lidar_points = PreprocessKittiDataset.read_point_cloud_data(self.point_cloud_file_path)
        pts = PreprocessKittiDataset.remove_points_outside_image_range(lidar_points, calib_dict["R0_rect"],
                                                                       calib_dict["Tr_velo_to_cam"],
                                                                       calib_dict["P2"],
                                                                       image_info_dict["image_shape"])

        if self.ground_truth_file_path is not None:
            annos_info = PreprocessKittiDataset.get_label_data_dict(self.ground_truth_file_path)
            annos_info = self._remove_dont_care(annos_info)
            annos_name = annos_info["name"]
            annos_location = annos_info["location"]
            annos_dimension = annos_info["dimensions"]
            rotation_y = annos_info["rotation_y"]
            gt_bboxes = np.concatenate([annos_location,
                                        annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
            tr_velo_to_cam = calib_dict["Tr_velo_to_cam"].astype(np.float32)
            r0_rect = calib_dict["R0_rect"].astype(np.float32)
            gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, tr_velo_to_cam, r0_rect)
            gt_labels = [self.object_category.get(name, -1) for name in annos_name]
            gt_bbox2d = annos_info["bbox"]
            data_dict = {
                "pts": pts,
                "gt_bboxes_3d": gt_bboxes_3d,
                "gt_labels": np.array(gt_labels),
                "gt_object_name": annos_name,
                "difficulty": annos_info["difficulty"],
                "image_info": image_info_dict,
                "calib_info": calib_dict,
                "gt_bbox2d": gt_bbox2d,
                "gt_bboxes_3d_camera": gt_bboxes
            }
        else:
            data_dict = {
                "pts": pts,
                "image_info": image_info_dict,
                "calib_info": calib_dict,
            }

        return self._get_points_within_lidar_range(data_dict)

    def _remove_dont_care(self, annos_info):
        keep_ids = [i for i, name in enumerate(annos_info["name"]) if name != "DontCare"]
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info

    def _get_points_within_lidar_range(self, data_dict):
        pts = data_dict["pts"]
        keep_mask = KITTI.mask_points_by_range(pts, self.pc_range)
        pts = pts[keep_mask]
        data_dict.update({"pts": pts})
        return data_dict

    def _load_data_gpu(self, data_dict):
        if self.use_cuda:
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].cuda()

    def _convert_data_dict(self, data_dict):
        if self.ground_truth_file_path is not None:
            model_input_dict = dict(
                batched_pts=[torch.from_numpy(data_dict["pts"])],
                batched_gt_bboxes=[torch.from_numpy(data_dict["gt_bboxes_3d"])],
                batched_labels=[torch.from_numpy(data_dict["gt_labels"])],
                batched_names=[data_dict["gt_object_name"]],
                batched_difficulty=[data_dict["difficulty"]],
                batched_img_info=[data_dict["image_info"]],
                batched_calib_info=[data_dict["calib_info"]]
            )  # follows the same format in the data loader
        else:
            model_input_dict = dict(
                batched_pts=[torch.from_numpy(data_dict["pts"])],
                batched_img_info=[data_dict["image_info"]],
                batched_calib_info=[data_dict["calib_info"]]
            )  # follows the same format in the data loader
        self._load_data_gpu(model_input_dict)
        return model_input_dict

    def _load_model(self):
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


if __name__ == "__main__":
    from config.kitti_config import cfg

    parser = argparse.ArgumentParser(description="Configuration Parameters")
    parser.add_argument("--point_cloud_file_path",
                        default=os.path.join(os.getcwd(), "Demo_dataset", "134", "000134.bin"),
                        help="the point cloud file path")
    parser.add_argument("--image_file_path",
                        default=os.path.join(os.getcwd(), "Demo_dataset", "134", "000134.png"),
                        help="the image file path")
    parser.add_argument("--calibration_file_path",
                        default=os.path.join(os.getcwd(), "Demo_dataset", "134", "000134_calib.txt"),
                        help="the calibration file path")
    parser.add_argument("--label_info_file_path",
                        default=os.path.join(os.getcwd(), "Demo_dataset", "134", "000134_label.txt"),
                        help="the ground truth/label_info_file_path path")
    args = parser.parse_args()

    infer = Inference(cfg, args.point_cloud_file_path, args.image_file_path,
                      args.calibration_file_path, args.label_info_file_path)
    infer()
