from collections import defaultdict
import numpy as np
import cv2
import os
import pickle
from tqdm import tqdm
from easydict import EasyDict
from utils.preprocess_fns import projection_matrix_to_CRT_kitti, get_frustum, points_camera2lidar, \
    group_rectangle_vertexs, group_plane_equation, points_in_bboxes, get_points_in_bboxes, get_points_num_in_bbox


# modified from: https://github.com/zhulf0804/PointPillars/blob/main/pre_process_kitti.py


class PreprocessKittiDataset:
    def __init__(self, config_file):
        self.root_path = config_file.root_path

    def create_data_info_pkl(self, split="train"):
        assert split in {"train", "val", "trainval"}
        kitti_dataset_info = dict()
        data_root = os.path.join(self.root_path, "dataset", "kitti")
        saved_path = os.path.join(data_root, "kitti_dataset_infos_" + split + ".pkl")
        if os.path.exists(saved_path):
            kitti_dataset_info = self.read_pickle_file(saved_path)
        else:
            ids_file = os.path.join(self.root_path, "dataset", "ImageSets", f"{split}.txt")
            with open(ids_file, "r") as f:
                file_ids = [file_id.strip() for file_id in f.readlines()]

            folder = "training" if split in {"train", "val", "trainval"} else "testing"
            build_gt_data_base = False
            if split in {"train", "trainval"}:
                build_gt_data_base = True
            if build_gt_data_base:
                kitti_gt_database = defaultdict(list)
                db_points_saved_path = os.path.join(data_root, "kitti_gt_database")
                os.makedirs(db_points_saved_path, exist_ok=True)

            for file_id in tqdm(file_ids):
                cur_info_dict = {}

                # get image info
                image_path = os.path.join(data_root, folder, "image_2", f"{file_id}.png")
                cur_info_dict["image"] = self.get_img_info(file_id, image_path)

                # get calibration info
                calib_path = os.path.join(data_root, folder, "calib", f"{file_id}.txt")
                calib_dict = self.get_calib_data_dict(calib_path)
                cur_info_dict["calib"] = calib_dict

                # get lidar data info
                lidar_point_path = os.path.join(data_root, folder, "velodyne", f"{file_id}.bin")
                lidar_points = self.read_point_cloud_data(lidar_point_path)
                reduced_lidar_points = self.remove_points_outside_image_range(lidar_points, calib_dict["R0_rect"],
                                                                              calib_dict["Tr_velo_to_cam"],
                                                                              calib_dict["P2"],
                                                                              cur_info_dict["image"]["image_shape"])
                saved_reduced_path = os.path.join(data_root, folder, "velodyne_reduced")
                os.makedirs(saved_reduced_path, exist_ok=True)
                saved_reduced_points_name = os.path.join(saved_reduced_path, f"{file_id}.bin")
                cur_info_dict["point_cloud_path"] = saved_reduced_points_name
                self.write_point_cloud_data(reduced_lidar_points, saved_reduced_points_name)

                if folder == "training":
                    # get label annotation data
                    label_path = os.path.join(data_root, folder, "label_2", f"{file_id}.txt")
                    annotation_dict = self.get_label_data_dict(label_path)
                    annotation_dict["num_points_in_gt"] = get_points_num_in_bbox(
                        points=reduced_lidar_points,
                        r0_rect=calib_dict["R0_rect"],
                        tr_velo_to_cam=calib_dict["Tr_velo_to_cam"],
                        dimensions=annotation_dict["dimensions"],
                        location=annotation_dict["location"],
                        rotation_y=annotation_dict["rotation_y"],
                        name=annotation_dict["name"])
                    cur_info_dict["annotation"] = annotation_dict

                    if build_gt_data_base:
                        # get ground truth database info
                        indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
                            get_points_in_bboxes(
                                points=lidar_points,
                                r0_rect=calib_dict["R0_rect"].astype(np.float32),
                                tr_velo_to_cam=calib_dict["Tr_velo_to_cam"].astype(np.float32),
                                dimensions=annotation_dict["dimensions"].astype(np.float32),
                                location=annotation_dict["location"].astype(np.float32),
                                rotation_y=annotation_dict["rotation_y"].astype(np.float32),
                                name=annotation_dict["name"]
                            )
                        for j in range(n_valid_bbox):
                            db_points = lidar_points[indices[:, j]]
                            db_points[:, :3] -= bboxes_lidar[j, :3]
                            db_points_saved_name = os.path.join(db_points_saved_path,
                                                                f"{int(file_id)}_{name[j]}_{j}.bin")
                            self.write_point_cloud_data(db_points, db_points_saved_name)

                            db_info = {
                                "name": name[j],
                                "path": os.path.join(os.path.basename(db_points_saved_path),
                                                     f"{int(file_id)}_{name[j]}_{j}.bin"),
                                "box3d_lidar": bboxes_lidar[j],
                                "difficulty": annotation_dict["difficulty"][j],
                                "num_points_in_gt": len(db_points),
                            }
                            if name[j] not in kitti_gt_database:
                                kitti_gt_database[name[j]] = [db_info]
                            else:
                                kitti_gt_database[name[j]].append(db_info)

                kitti_dataset_info[int(file_id)] = cur_info_dict

            self.write_pickle(kitti_dataset_info, saved_path)
            if build_gt_data_base:
                saved_db_path = os.path.join(data_root, "kitti_dbinfos_train.pkl")
                self.write_pickle(kitti_gt_database, saved_db_path)

        return kitti_dataset_info

    @staticmethod
    def read_pickle_file(file_path):
        assert os.path.splitext(file_path)[1] == ".pkl"
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def write_pickle(data, save_path):
        assert os.path.splitext(save_path)[1] == ".pkl"
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def read_point_cloud_data(file_path, dim=4):
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)

    @staticmethod
    def write_point_cloud_data(points, file_path):
        with open(file_path, "w") as f:
            points.tofile(f)

    @staticmethod
    def get_img_info(file_id, image_path):
        img = cv2.imread(image_path)
        image_shape = img.shape[:2]
        return {
            "image_path": image_path,
            "image_idx": int(file_id),
            "image_shape": image_shape
        }

    @staticmethod
    def get_calib_data_dict(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        calib_data = [line.strip() for line in lines]

        # get all related calibration data
        p0 = np.array([item for item in calib_data[0].split(" ")[1:]], dtype=np.float32).reshape(3, 4)
        p1 = np.array([item for item in calib_data[1].split(" ")[1:]], dtype=np.float32).reshape(3, 4)
        p2 = np.array([item for item in calib_data[2].split(" ")[1:]], dtype=np.float32).reshape(3, 4)
        p3 = np.array([item for item in calib_data[3].split(" ")[1:]], dtype=np.float32).reshape(3, 4)
        rect = np.array([item for item in calib_data[4].split(" ")[1:]], dtype=np.float32).reshape(3, 3)
        velo_to_cam = np.array([item for item in calib_data[5].split(" ")[1:]], dtype=np.float32).reshape(3, 4)

        p0 = np.concatenate([p0, np.array([[0, 0, 0, 1]])], axis=0)
        p1 = np.concatenate([p1, np.array([[0, 0, 0, 1]])], axis=0)
        p2 = np.concatenate([p2, np.array([[0, 0, 0, 1]])], axis=0)
        p3 = np.concatenate([p3, np.array([[0, 0, 0, 1]])], axis=0)
        rect_new = np.eye(4, dtype=rect.dtype)
        rect_new[:3, :3] = rect
        rect = rect_new
        velo_to_cam = np.concatenate([velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0)

        calib_data = EasyDict({
            "P0": p0,
            "P1": p1,
            "P2": p2,
            "P3": p3,
            "R0_rect": rect,
            "Tr_velo_to_cam": velo_to_cam,
        })
        return calib_data

    @staticmethod
    def get_label_data_dict(file_path):
        def judge_difficulty(annotation_dict):
            truncated = annotation_dict.truncated
            occluded = annotation_dict.occluded
            bbox = annotation_dict.bbox
            height = bbox[:, 3] - bbox[:, 1]

            MIN_HEIGHTS = [40, 25, 25]
            MAX_OCCLUSION = [0, 1, 2]
            MAX_TRUNCATION = [0.15, 0.30, 0.50]
            difficultys = []
            for h, o, t in zip(height, occluded, truncated):
                difficulty = -1
                for i in range(2, -1, -1):
                    if h > MIN_HEIGHTS[i] and o <= MAX_OCCLUSION[i] and t <= MAX_TRUNCATION[i]:
                        difficulty = i
                difficultys.append(difficulty)
            label_data.difficulty = np.array(difficultys, dtype=np.int32)

        with open(file_path, "r") as f:
            lines = f.readlines()
        labels = np.array([line.strip().split(" ") for line in lines], dtype=np.str_)
        label_data = EasyDict()
        label_data.name = labels[:, 0]
        label_data.truncated = labels[:, 1].astype(np.float32)
        label_data.occluded = labels[:, 2].astype(np.int32)
        label_data.alpha = labels[:, 3].astype(np.float32)
        label_data.bbox = labels[:, 4:8].astype(np.float32)
        label_data.dimensions = labels[:, 8:11].astype(np.float32)[:, [2, 0, 1]]  # [hwl represent yzx axis]
        label_data.location = labels[:, 11:14].astype(np.float32)  # [x, y, z]
        label_data.rotation_y = labels[:, 14].astype(np.float32)
        judge_difficulty(label_data)

        # delete "don"t care" label data
        keep_ids = [i for i, class_name in enumerate(label_data.name) if class_name != "DontCare"]
        for k, v in label_data.items():
            label_data[k] = v[keep_ids]
        return label_data

    @staticmethod
    def remove_points_outside_image_range(points, r0_rect, tr_velo_to_cam, P2, image_shape):
        C, R, T = projection_matrix_to_CRT_kitti(P2)
        image_bbox = [0, 0, image_shape[1], image_shape[0]]
        frustum = get_frustum(image_bbox, C)
        frustum -= T
        frustum = np.linalg.inv(R) @ frustum.T
        frustum = points_camera2lidar(frustum.T[None, ...], tr_velo_to_cam, r0_rect)  # (1, 8, 3)
        group_rectangle_vertexs_v = group_rectangle_vertexs(frustum)
        frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
        indices = points_in_bboxes(points[:, :3], frustum_surfaces)  # (N, 1)
        points = points[indices.reshape([-1])]
        return points
