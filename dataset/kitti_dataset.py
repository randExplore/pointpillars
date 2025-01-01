import os
import numpy as np
import pickle
from .get_data_info import PreprocessKittiDataset
from torch.utils.data import Dataset
from PIL import Image
from utils.preprocess_fns import bbox_camera2lidar, limit_period
from .data_augmentation import data_augment, BaseSampler

# modified from: https://github.com/zhulf0804/PointPillars/blob/main/dataset/kitti.py


class KITTI(Dataset):
    def __init__(self, cfg, split=None):
        self.object_category = cfg.class_name
        self.pc_range = cfg.pc_range
        self.voxel_size = cfg.voxel_size
        self.max_num_points = cfg.max_num_points
        self.max_voxels = cfg.max_voxels[0]
        self.point_dim = cfg.point_dim
        self.data_aug_option = cfg.data_augmentaion
        self.include_img = cfg.include_img_data

        # Read dataset info
        data_preprocessor = PreprocessKittiDataset(cfg)
        if split is None:
            self.data_infos = data_preprocessor.create_data_info_pkl(cfg.split)
            self.split = cfg.split
        else:
            self.data_infos = data_preprocessor.create_data_info_pkl(split)
            self.split = split

        if self.split == "test":
            raise ValueError("The dataset file only supports {train, val, trainval} splits. "
                             "Please use inference.py file for testing split dataset!")

        self.sorted_index = sorted(list(self.data_infos.keys()))
        self.data_root = os.path.join(cfg.root_path, "dataset", "kitti")

        db_infos = self.read_pickle(os.path.join(self.data_root, "kitti_dbinfos_train.pkl"))
        db_infos = self.filter_db(db_infos)

        db_sampler = {}
        for cat_name in self.object_category:
            db_sampler[cat_name] = BaseSampler(db_infos[cat_name], shuffle=True)
        self.data_aug_config = dict(
            db_sampler=dict(
                db_sampler=db_sampler,
                sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10)
            ),
            object_noise=dict(
                num_try=100,
                translation_std=[0.25, 0.25, 0.25],
                rot_range=[-0.15707963267, 0.15707963267]
            ),
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
            )
        )

    def remove_dont_care(self, annos_info):
        keep_ids = [i for i, name in enumerate(annos_info["name"]) if name != "DontCare"]
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info

    def filter_db(self, db_infos):
        # filter_by_difficulty
        for k, v in db_infos.items():
            db_infos[k] = [item for item in v if item["difficulty"] != -1]

        # filter_by_min_points
        filter_thrs = dict(Car=5, Pedestrian=10, Cyclist=10)
        for cat in self.object_category:
            filter_thr = filter_thrs[cat]
            db_infos[cat] = [item for item in db_infos[cat] if item["num_points_in_gt"] >= filter_thr]

        return db_infos

    def __getitem__(self, index):
        data_info = self.data_infos[self.sorted_index[index]]
        image_info, calib_info, annos_info = \
            data_info["image"], data_info["calib"], data_info["annotation"]

        pts = PreprocessKittiDataset.read_point_cloud_data(data_info["point_cloud_path"])

        # annotations input
        annos_info = self.remove_dont_care(annos_info)
        annos_name = annos_info["name"]
        annos_location = annos_info["location"]
        annos_dimension = annos_info["dimensions"]
        rotation_y = annos_info["rotation_y"]
        gt_bboxes = np.concatenate([annos_location,
                                    annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
        tr_velo_to_cam = calib_info["Tr_velo_to_cam"].astype(np.float32)
        r0_rect = calib_info["R0_rect"].astype(np.float32)
        gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, tr_velo_to_cam, r0_rect)
        gt_labels = [self.object_category.get(name, -1) for name in annos_name]
        gt_bbox2d = annos_info["bbox"]
        data_dict = {
            "pts": pts,
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels": np.array(gt_labels),
            "gt_object_name": annos_name,
            "difficulty": annos_info["difficulty"],
            "image_info": image_info,
            "calib_info": calib_info,
            "gt_bbox2d": gt_bbox2d,
            "gt_bboxes_3d_camera": gt_bboxes
        }
        if self.split in ["train", "trainval"]:
            data_dict = data_augment(self.object_category, self.data_root, data_dict, self.data_aug_config)
            data_dict = self.get_points_within_lidar_range(data_dict)
            data_dict = self.get_object_within_lidar_range(data_dict)
            data_dict = self.shuffle_point_cloud_data(data_dict)
        else:
            data_dict = self.get_points_within_lidar_range(data_dict)
        if self.include_img:
            data_dict["image"] = Image.open(os.path.join(self.data_root, image_info["image_path"]))
        return data_dict

    def __len__(self):
        return len(self.data_infos)

    def get_points_within_lidar_range(self, data_dict):
        pts = data_dict["pts"]
        keep_mask = self.mask_points_by_range(pts, self.pc_range)
        pts = pts[keep_mask]
        data_dict.update({"pts": pts})
        return data_dict

    def get_object_within_lidar_range(self, data_dict):
        gt_bboxes_3d, gt_labels = data_dict["gt_bboxes_3d"], data_dict["gt_labels"]
        gt_names, difficulty = data_dict["gt_object_name"], data_dict["difficulty"]
        gt_bboxes_3d_camera = data_dict["gt_bboxes_3d_camera"]

        # use BEV
        flag_x_low = gt_bboxes_3d[:, 0] > self.pc_range[0]
        flag_y_low = gt_bboxes_3d[:, 1] > self.pc_range[1]
        flag_x_high = gt_bboxes_3d[:, 0] < self.pc_range[3]
        flag_y_high = gt_bboxes_3d[:, 1] < self.pc_range[4]
        keep_mask = flag_x_low & flag_y_low & flag_x_high & flag_y_high

        gt_bboxes_3d, gt_labels = gt_bboxes_3d[keep_mask], gt_labels[keep_mask]
        if not self.data_aug_option:
            gt_bboxes_3d_camera = gt_bboxes_3d_camera[keep_mask]
        gt_names, difficulty = gt_names[keep_mask], difficulty[keep_mask]
        gt_bboxes_3d[:, 6] = limit_period(gt_bboxes_3d[:, 6], 0.5, 2 * np.pi)
        data_dict.update({"gt_bboxes_3d": gt_bboxes_3d})
        data_dict.update({"gt_labels": gt_labels})
        data_dict.update({"gt_object_name": gt_names})
        data_dict.update({"difficulty": difficulty})
        data_dict.update({"gt_bboxes_3d_camera": gt_bboxes_3d_camera})
        return data_dict

    @staticmethod
    def shuffle_point_cloud_data(data_dict):
        pts = data_dict["pts"]
        indices = np.arange(0, len(pts))
        np.random.shuffle(indices)
        pts = pts[indices]
        data_dict.update({"pts": pts})
        return data_dict

    @staticmethod
    def read_pickle(file_path, suffix=".pkl"):
        assert os.path.splitext(file_path)[1] == suffix
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def mask_points_by_range(points, limit_range):
        mask = ((points[:, 0] > limit_range[0]) & (points[:, 0] < limit_range[3]) & (points[:, 1] > limit_range[1]) & (
                    points[:, 1] < limit_range[4]) & (points[:, 2] > limit_range[2]) & (points[:, 2] < limit_range[5]))
        return mask
