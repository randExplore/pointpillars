import numpy as np
import cv2
import open3d as o3d
import os
import torch

from .nms_cal import boxes_to_corners_3d, boxes_camera_to_corners_3d
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion
import matplotlib.pyplot as plt


# modified from:
# https://github.com/zhulf0804/PointPillars/blob/main/utils/vis_o3d.py


# {Pedestrian: red, Cyclist: green, Car: blue, Ground Truth: yellow}
OBJECT_NAMES = {"Pedestrian": 0, "Cyclist": 1, "Car": 2, "Ground_truth": 3}
LABEL_TO_OBJECT_NAMES = {0: "Pedestrian", 1: "Cyclist", 2: "Car"}
COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
COLORS_IMG_BGR = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]]

LINES = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [2, 6],
    [7, 3],
    [1, 5],
    [4, 0]
]


def bbox_lidar2camera(bboxes, tr_velo_to_cam, r0_rect):
    """
    bboxes: shape=(N, 7)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 7)
    """
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([y_size, z_size, x_size], axis=1)
    extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), "constant", constant_values=1.0)
    rt_mat = r0_rect @ tr_velo_to_cam
    xyz = extended_xyz @ rt_mat.T
    bboxes_camera = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return bboxes_camera


def points_camera2image(points, P2):
    """
    points: shape=(N, 8, 3)
    P2: shape=(4, 4)
    return: shape=(N, 8, 2)
    """
    extended_points = np.pad(points, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1.0)  # (n, 8, 4)
    image_points = extended_points @ P2.T  # (N, 8, 4)
    image_points = image_points[:, :, :2] / image_points[:, :, 2:3]
    return image_points


def points_lidar2image(points, tr_velo_to_cam, r0_rect, P2):
    """
    points: shape=(N, 8, 3)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    P2: shape=(4, 4)
    return: shape=(N, 8, 2)
    """
    extended_points = np.pad(points, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1.0)  # (N, 8, 4)
    rt_mat = r0_rect @ tr_velo_to_cam
    camera_points = extended_points @ rt_mat.T  # (N, 8, 4)
    image_points = camera_points @ P2.T  # (N, 8, 4)
    image_points = image_points[:, :, :2] / image_points[:, :, 2:3]

    return image_points


def points_camera2lidar(points, tr_velo_to_cam, r0_rect):
    """
    points: shape=(N, 8, 3)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 8, 3)
    """
    extended_xyz = np.pad(points, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    return xyz[..., :3]


def filter_bbox_in_image_range(result, tr_velo_to_cam, r0_rect, P2, image_shape):
    """
    result: dict(bboxes3D, labels, scores)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    P2: shape=(4, 4)
    image_shape: (h, w)
    return: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    """
    h, w = image_shape

    bboxes3D = result["bboxes3D"]
    labels = result["labels"]
    scores = result["scores"]
    camera_bboxes = bbox_lidar2camera(bboxes3D, tr_velo_to_cam, r0_rect)  # (n, 7)
    bboxes_points = boxes_camera_to_corners_3d(camera_bboxes)  # (n, 8, 3)
    image_points = points_camera2image(bboxes_points, P2)  # (n, 8, 2)
    image_x1y1 = np.min(image_points, axis=1)  # (n, 2)
    image_x1y1 = np.maximum(image_x1y1, 0)
    image_x2y2 = np.max(image_points, axis=1)  # (n, 2)
    image_x2y2 = np.minimum(image_x2y2, [w, h])
    bboxes2d = np.concatenate([image_x1y1, image_x2y2], axis=-1)

    keep_flag = (image_x1y1[:, 0] < w) & (image_x1y1[:, 1] < h) & (image_x2y2[:, 0] > 0) & (image_x2y2[:, 1] > 0)

    result = {
        "bboxes3D": bboxes3D[keep_flag],
        "labels": labels[keep_flag],
        "scores": scores[keep_flag],
        "bboxes2d": bboxes2d[keep_flag],
        "camera_bboxes": camera_bboxes[keep_flag]
    }
    return result


def filter_bbox_in_lidar_range(result, pcd_limit_range):
    """
    result: dict(bboxes3D, labels, scores, bboxes2d, camera_bboxes)
    pcd_limit_range: []
    return: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    """
    lidar_bboxes, labels, scores = result["bboxes3D"], result["labels"], result["scores"]
    if "bboxes2d" not in result:
        result["bboxes2d"] = np.zeros_like(lidar_bboxes[:, :4])
    if "camera_bboxes" not in result:
        result["camera_bboxes"] = np.zeros_like(lidar_bboxes)
    bboxes2d, camera_bboxes = result["bboxes2d"], result["camera_bboxes"]
    flag1 = lidar_bboxes[:, :3] > pcd_limit_range[:3][None, :]  # (n, 3)
    flag2 = lidar_bboxes[:, :3] < pcd_limit_range[3:][None, :]  # (n, 3)
    keep_flag = np.all(flag1, axis=-1) & np.all(flag2, axis=-1)

    result = {
        "bboxes3D": lidar_bboxes[keep_flag],
        "labels": labels[keep_flag],
        "scores": scores[keep_flag],
        "bboxes2d": bboxes2d[keep_flag],
        "camera_bboxes": camera_bboxes[keep_flag]
    }
    return result


def npy2ply(npy):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(npy[:, :3])
    density = npy[:, 3]
    colors = [[item, item, item] for item in density]
    ply.colors = o3d.utility.Vector3dVector(colors)
    return ply


def add_bbox_3d(points, color=[1, 0, 0]):
    colors = [color for i in range(len(LINES))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(LINES),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def plot_point_cloud(data, file_id=None, save=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for obj in data:
        vis.add_geometry(obj)
        vis.update_geometry(obj)
    vis.poll_events()
    vis.update_renderer()

    if save:
        save_path = os.path.join(os.getcwd(), "Demo_dataset", "prediction")
        os.makedirs(save_path, exist_ok=True)
        img_save_path = os.path.join(save_path, "Lidar_pred_" + file_id + ".png")
        vis.capture_screen_image(img_save_path)
    vis.run()
    vis.destroy_window()


def plot_point_cloud_data(pc, bboxes=None, labels=None, file_id=None, save=False):
    def add_text_3d(text, pos, color, direction=None, degree=0.0, density=10,
            font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', font_size=10):
        """
        Generate a 3D text point cloud used for visualization.
        :param text: content of the text
        :param pos: 3D xyz position of the text upper left corner
        :param color: text color
        :param direction: 3D normalized direction of where the text faces
        :param degree: in plane rotation of text
        :param font: Name of the font - change it according to your system
        :param font_size: size of the font
        :return: o3d.geoemtry.PointCloud object
        """
        if direction is None:
            direction = (0., 0., 1.)

        font_obj = ImageFont.truetype(font, font_size * density)
        font_dim = font_obj.getbbox(text)[-2:]

        img = Image.new('RGB', font_dim, color="white")
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font_obj, fill=color)
        img = img.rotate(270, expand=True)
        img = np.asarray(img)
        img_mask = img[:, :, 0] < 256
        indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
        pcd.points = o3d.utility.Vector3dVector(indices / 10 / density)

        raxis = np.cross([0.0, 0.0, 1.0], direction)
        if np.linalg.norm(raxis) < 1e-6:
            raxis = (0.0, 0.0, 1.0)
        trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
                 Quaternion(axis=direction, degrees=degree)).transformation_matrix
        trans[0:3, 3] = np.asarray(pos)
        pcd.transform(trans)
        return pcd

    if save and file_id is None:
        raise ValueError("The file_id name string can't be None if you want to save the point cloud results!")

    if isinstance(pc, np.ndarray):
        pc = npy2ply(pc)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0, 0, 0])

    if bboxes is None:
        plot_point_cloud([pc, coordinate_frame], file_id, save)
        return

    if len(bboxes.shape) == 2:
        bboxes = boxes_to_corners_3d(bboxes)

    vis_objs = [pc, coordinate_frame]
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        if labels is None:
            color = [1, 1, 0]
        else:
            if 0 <= labels[i] < 3:
                color = COLORS[labels[i]]
            else:
                color = COLORS[-1]
        vis_objs.append(add_bbox_3d(bbox, color=color))

    #  Add object labels with their own color
    if all([label == -1 for label in labels]):
        object_name = "Ground_truth"
        color_idx = OBJECT_NAMES[object_name]
        text_obj = add_text_3d(object_name, pos=[25, 16, 5], color=tuple(COLORS_IMG_BGR[color_idx][::-1]))
        vis_objs.append(text_obj)
    else:
        for i, object_name in enumerate(OBJECT_NAMES):
            color_idx = OBJECT_NAMES[object_name]
            text_obj = add_text_3d(object_name, pos=[25, 20-2*i, 5], color=tuple(COLORS_IMG_BGR[color_idx][::-1]))
            vis_objs.append(text_obj)

    plot_point_cloud(vis_objs, file_id, save)


def plot_img_3d_bboxes(img, image_points, labels):
    def add_label_text(image, coordinate, text, color):
        """Draws the label on its bounding box"""
        x, y = coordinate
        cv2.putText(image, text, (int(x), int(y) - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    for i in range(len(image_points)):
        label = labels[i]
        bbox_points = image_points[i]  # (8, 2)
        if 0 <= label < 3:
            color = COLORS_IMG_BGR[label]
            add_label_text(img, bbox_points[4], LABEL_TO_OBJECT_NAMES[label], color)
        else:
            color = COLORS_IMG_BGR[-1]
        for line_id in LINES:
            x1, y1 = bbox_points[line_id[0]]
            x2, y2 = bbox_points[line_id[1]]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(img, (x1, y1), (x2, y2), color, 1)
    return img


def draw_pred_bboxes_on_img(calib_info, img, pred_res, labels):
    tr_velo_to_cam = calib_info["Tr_velo_to_cam"].astype(np.float32)
    r0_rect = calib_info["R0_rect"].astype(np.float32)
    P2 = calib_info["P2"].astype(np.float32)

    image_shape = img.shape[:2]
    pred_res = filter_bbox_in_image_range(pred_res, tr_velo_to_cam, r0_rect, P2, image_shape)
    bboxes2d, camera_bboxes = pred_res["bboxes2d"], pred_res["camera_bboxes"]
    bboxes_corners = boxes_camera_to_corners_3d(camera_bboxes)
    image_points = points_camera2image(bboxes_corners, P2)
    img = plot_img_3d_bboxes(img, image_points, labels)
    return img


def plot_all(pred_res, gt_dict, save_pred_result=False):
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)
    pc = gt_dict["pts"]
    if "calib_info" in gt_dict:
        calib_info = gt_dict["calib_info"]
    else:
        calib_info = None

    if "gt_labels" in gt_dict:
        gt_labels = gt_dict["gt_labels"]
    else:
        gt_labels = None

    if "image_path" in gt_dict["image_info"]:
        img = cv2.imread(gt_dict["image_info"]["image_path"], 1)
    else:
        img = None

    if "gt_bboxes_3d" in gt_dict:
        gt_bboxes = gt_dict["gt_bboxes_3d"]
    else:
        gt_bboxes = None

    if "gt_bboxes_3d_camera" in gt_dict:
        gt_bboxes_camera = gt_dict["gt_bboxes_3d_camera"]
    else:
        gt_bboxes_camera = None

    no_gt_flag = (gt_bboxes_camera is None or gt_labels is None or gt_bboxes is None)
    pred_res = filter_bbox_in_lidar_range(pred_res, pcd_limit_range)
    bboxes3D = pred_res["bboxes3D"]
    labels, scores = pred_res["labels"], pred_res["scores"]
    file_name = os.path.splitext(os.path.basename(gt_dict["image_info"]["image_path"]))[0]

    if calib_info is None:
        plot_point_cloud_data(pc, bboxes3D, labels, file_name, save_pred_result)
    else:
        if no_gt_flag:
            if img is not None:
                plot_point_cloud_data(pc, bboxes3D, labels, file_name, save_pred_result)
                img = draw_pred_bboxes_on_img(calib_info, img, pred_res, labels)
        else:
            virtual_labels = [-1] * gt_labels.shape[0]
            pred_gt_lidar_bboxes = np.concatenate([bboxes3D, gt_bboxes], axis=0)
            pred_gt_labels = np.concatenate([labels, virtual_labels])
            plot_point_cloud_data(pc, pred_gt_lidar_bboxes, pred_gt_labels, file_name, save_pred_result)

            if img is not None:
                # plot prediction first
                img = draw_pred_bboxes_on_img(calib_info, img, pred_res, labels)

                # plot ground truth
                P2 = calib_info["P2"].astype(np.float32)
                bboxes_corners = boxes_camera_to_corners_3d(gt_bboxes_camera)
                image_points = points_camera2image(bboxes_corners, P2)
                img = plot_img_3d_bboxes(img, image_points, virtual_labels)

        if img is not None:
            if save_pred_result:
                save_path = os.path.join(os.getcwd(), "Demo_dataset", "prediction")
                os.makedirs(save_path, exist_ok=True)
                filename = os.path.join(save_path, "image_pred_3dbbox_" + file_name + ".png")
                cv2.imwrite(filename, img)
            cv2.imshow(f"{os.path.basename(gt_dict['image_info']['image_path'])}-3d bbox", img)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()


def plot_tensorboard(lr, momentum, train_loss, val_loss, save_path=None, save_fig=False):
    fig, ax = plt.subplots(2, 1)
    fig.suptitle("Learning rate and momentum", fontsize=12)
    ax[0].plot(lr[:, 0], lr[:, 1])
    ax[0].set_xlabel("Global training steps")
    ax[0].set_ylabel("Learning rate")
    ax[0].grid()

    ax[1].plot(momentum[:, 0], momentum[:, 1])
    ax[1].set_xlabel("Global training steps")
    ax[1].set_ylabel("Momentum")
    ax[1].grid()
    fig.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(save_path, "train_lr_momentum.png"))
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    fig.suptitle("Training and validation loss", fontsize=12)
    ax.plot(train_loss[:, 0], train_loss[:, 1], label="train_loss")
    ax.plot(val_loss[:, 0], val_loss[:, 1], label="val_loss")
    ax.set_xlabel("Global training steps")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(save_path, "train_loss.png"))
    plt.show()
    plt.close()
