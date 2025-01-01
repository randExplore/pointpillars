import torch
import copy
from utils.preprocess_fns import limit_period
import numpy as np

# modified from: https://github.com/zhulf0804/PointPillars/blob/main/model/anchors.py


class AnchorBoxes:
    def __init__(self,anchor_boxes_cfg, pc_range):
        self.anchor_range = pc_range
        self.anchor_boxes_cfg = anchor_boxes_cfg
        self.anchor_sizes = [config["anchor_sizes"][0] for config in anchor_boxes_cfg]
        self.anchor_rotations = [config["anchor_rotations"] for config in anchor_boxes_cfg]
        self.anchor_heights = [config["anchor_bottom_heights"] for config in anchor_boxes_cfg]
        self.align_center = [config.get("align_center", False) for config in anchor_boxes_cfg]

    def generate_anchor_boxes(self, feature_map_size, device):
        """
        generate_anchor_boxes for all object classes
        :param feature_map_size: [y_dim, x_dim]
        :return: generated anchor boxes, shape is [x_dim, y_dim, num_object_class, num_rotations, 7]
        """
        num_object_class = torch.tensor(len(self.anchor_boxes_cfg), device=device)
        anchor_sizes = torch.tensor(self.anchor_sizes, device=device)
        rotations = torch.tensor(self.anchor_rotations, device=device)
        anchor_boxes= []
        for i in range(num_object_class):
            cur_anchor_size = anchor_sizes[i]
            cur_rotation = rotations[i]
            cur_z_bottom_height = self.anchor_heights[i]
            x_centers = torch.linspace(self.anchor_range[0], self.anchor_range[3],
                                       feature_map_size[1] + 1, device=device)
            y_centers = torch.linspace(self.anchor_range[1], self.anchor_range[4],
                                       feature_map_size[0] + 1, device=device)

            x_shift = (x_centers[1] - x_centers[0]) / 2
            y_shift = (y_centers[1] - y_centers[0]) / 2

            # shift to centers
            x_centers = x_centers[:feature_map_size[1]] + x_shift
            y_centers = y_centers[:feature_map_size[0]] + y_shift
            z_centers = torch.tensor(cur_z_bottom_height, device=device)

            # [feature_map_size[1], feature_map_size[0], 1, 2] * 4
            mesh_grids = list(torch.meshgrid(x_centers, y_centers, z_centers, cur_rotation, indexing="ij"))
            for j in range(len(mesh_grids)):
                mesh_grids[j] = mesh_grids[j][..., None]  # [feature_map_size[1], feature_map_size[0], 1, 2, 1]

            anchor_size = cur_anchor_size[None, None, None, None, :]
            repeat_shape = [feature_map_size[1], feature_map_size[0], 1, len(cur_rotation), 1]
            anchor_size = anchor_size.repeat(repeat_shape)  # [feature_map_size[1], feature_map_size[0], 1, 2, 3]
            mesh_grids.insert(3, anchor_size)

            # anchors shape is [feature_map_size[0], feature_map_size[1], 1, 2, 7]
            anchors = torch.cat(mesh_grids,
                                dim=-1).permute(2, 1,
                                                0, 3, 4).contiguous().squeeze(0)  # [y_dim, x_dim, 1, 2, 7]
            anchors = anchors[:, :, None, :, :]  # prepare the new axis for the num_object_class dim
            anchor_boxes.append(anchors)

        # anchor_boxes shape is [y_dim, x_dim, num_object_class, 2, 7], 7 dim represents [x, y, z, l, h, w, angle]
        anchor_boxes = torch.cat(anchor_boxes, dim=2)
        return anchor_boxes


def bboxes2xycorners(bboxes):
    """
    bboxes: (n, 7), (x, y, z, w, l, h, angle)
    return: (n, 4), (x1, y1, x2, y2)
    """
    bboxes_bev = copy.deepcopy(bboxes[:, [0, 1, 3, 4]])
    bboxes_angle = limit_period(bboxes[:, 6].cpu(), offset=0.5, period=np.pi).to(bboxes_bev.device)
    bboxes_bev = torch.where(torch.abs(bboxes_angle[:, None]) > np.pi / 4, bboxes_bev[:, [0, 1, 3, 2]], bboxes_bev)
    bboxes_xy = bboxes_bev[:, :2]
    bboxes_wl = bboxes_bev[:, 2:]
    bboxes_bev_x1y1x2y2 = torch.cat([bboxes_xy - bboxes_wl / 2, bboxes_xy + bboxes_wl / 2], dim=-1)
    return bboxes_bev_x1y1x2y2


def iou2d(bboxes1, bboxes2):
    bboxes1_bev = bboxes2xycorners(bboxes1)  # [n, 4]
    bboxes2_bev = bboxes2xycorners(bboxes2)  # [m, 4]

    # calculate the iou of the above two 2d boxes
    bboxes_x1 = torch.maximum(bboxes1_bev[:, 0][:, None], bboxes2_bev[:, 0][None, :])  # [n, m]
    bboxes_y1 = torch.maximum(bboxes1_bev[:, 1][:, None], bboxes2_bev[:, 1][None, :])  # [n, m]
    bboxes_x2 = torch.minimum(bboxes1_bev[:, 2][:, None], bboxes2_bev[:, 2][None, :])
    bboxes_y2 = torch.minimum(bboxes1_bev[:, 3][:, None], bboxes2_bev[:, 3][None, :])

    bboxes_w = torch.clamp(bboxes_x2 - bboxes_x1, min=0)
    bboxes_h = torch.clamp(bboxes_y2 - bboxes_y1, min=0)
    iou_area = bboxes_w * bboxes_h  # [n, m]

    bboxes1_wh = bboxes1_bev[:, 2:] - bboxes1_bev[:, :2]
    area1 = bboxes1_wh[:, 0] * bboxes1_wh[:, 1]
    bboxes2_wh = bboxes2_bev[:, 2:] - bboxes2_bev[:, :2]
    area2 = bboxes2_wh[:, 0] * bboxes2_wh[:, 1]
    iou = iou_area / (area1[:, None] + area2[None, :] - iou_area + 1e-8)  # [n, m]
    return iou