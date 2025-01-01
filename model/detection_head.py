import torch
from torch import nn
import numpy as np
from easydict import EasyDict
from .anchor_boxes import AnchorBoxes, iou2d
from utils.preprocess_fns import limit_period
from utils.nms_cal import nms_3d

# modified and referenced from:
# https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/task_modules/assigners/max_3d_iou_assigner.py
# https://github.com/zhulf0804/PointPillars/blob/main/model/pointpillars.py


class DetectionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mode = cfg.mode
        self.pc_range = np.array(cfg.pc_range)
        self.num_dir_bins = cfg.num_dir_bins
        self.num_object_classes = len(cfg.class_name)
        self.num_anchor_boxes_per_location = self.num_dir_bins * self.num_object_classes
        self.anchors_generator = AnchorBoxes(cfg.anchor_boxes_cfg, self.pc_range)
        self.anchor_boxes_cfg = cfg.anchor_boxes_cfg

        self.nms_pre_maxsize = cfg.nms_cfg.nms_pre_maxsize
        self.nms_thr = cfg.nms_cfg.nms_th
        self.score_thr = cfg.score_thr
        self.nms_post_maxsize = cfg.nms_cfg.nms_post_maxsize

        input_channels = cfg.input_channels
        self.conv_cls = nn.Conv2d(input_channels,
                                  self.num_anchor_boxes_per_location * self.num_object_classes,
                                  kernel_size=1)
        self.conv_reg = nn.Conv2d(input_channels,
                                      self.num_anchor_boxes_per_location * 7,  # 7 here means [x, y, z, w, l, h, angle]
                                      kernel_size=1)

        self.conv_dir_cls = nn.Conv2d(input_channels,
                                      self.num_anchor_boxes_per_location * self.num_dir_bins,
                                      kernel_size=1)
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x, batch_size, batched_data_dict):
        bbox_cls_pred = self.conv_cls(x)  # [batch_size, n_anchors*3, 248, 216]
        bbox_pred = self.conv_reg(x)  # [batch_size, n_anchors*7, 248, 216]
        bbox_dir_cls_pred = self.conv_dir_cls(x)  # [batch_size, n_anchors*2, 248, 216]

        device = bbox_cls_pred.device
        feature_map_size = list(bbox_cls_pred.size()[-2:])
        anchor_boxes = self.anchors_generator.generate_anchor_boxes(feature_map_size, device)
        batch_anchors = [anchor_boxes for _ in range(batch_size)]

        if self.mode != "training":
            return self.generate_batched_pred_bbox(batch_anchors, bbox_cls_pred,
                                                   bbox_pred, bbox_dir_cls_pred)
        else:
            return self.generate_predictions_for_train(batched_data_dict, batch_anchors,
                                                       bbox_cls_pred, bbox_pred, bbox_dir_cls_pred)

    @torch.no_grad()
    def generate_batched_pred_bbox(self, batch_anchors, batch_bbox_cls_pred,
                                   batch_bbox_pred, batch_bbox_dir_cls_pred):
        batch_pred_bboxes = []
        batch_size = len(batch_anchors)
        for i in range(batch_size):
            bbox_cls_pred = batch_bbox_cls_pred[i].permute(1, 2, 0).reshape(-1, self.num_object_classes)
            bbox_pred = batch_bbox_pred[i].permute(1, 2, 0).reshape(-1, 7)
            bbox_dir_cls_pred = batch_bbox_dir_cls_pred[i].permute(1, 2, 0).reshape(-1, 2)
            anchors = batch_anchors[i].reshape(-1, 7)

            bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
            bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

            # Filter to find the predicted bboxes that have scores larger than nms_pre
            index = bbox_cls_pred.max(1)[0].topk(self.nms_pre_maxsize)[1]
            bbox_cls_pred = bbox_cls_pred[index]
            bbox_pred = bbox_pred[index]
            bbox_dir_cls_pred = bbox_dir_cls_pred[index]
            anchors = anchors[index]

            # convert to offset the predicted bboxes format with anchors since the predicted bbox is under this format
            # [dx, dy, dz, dw, dl, dh, dtheta]
            bbox_pred = self.convert_anchors2bboxes(anchors, bbox_pred)
            bbox_pred2d_xy = bbox_pred[:, [0, 1]]
            bbox_pred2d_lw = bbox_pred[:, [3, 4]]
            bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                     bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                     bbox_pred[:, 6:]], dim=-1)  # (n_anchors, 5)

            # perform the non-maximum suppression process which are based on each object class
            bbox_pred_final, bbox_pred_label_final, bbox_cls_score_final = [], [], []
            for j in range(self.num_object_classes):
                # first filter out those bboxes with scores below self.score_thr
                cur_bbox_cls_pred = bbox_cls_pred[:, j]
                score_inds = cur_bbox_cls_pred > self.score_thr
                if score_inds.sum() == 0:
                    continue
                cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
                cur_bbox_pred = bbox_pred[score_inds]
                cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]

                # calculate nms scores
                select_index = nms_3d(
                    cur_bbox_pred.detach().cpu().numpy(), cur_bbox_cls_pred.detach().cpu().numpy(),
                    nms_thres=self.nms_thr, score_thres=self.score_thr
                )

                cur_bbox_cls_pred = cur_bbox_cls_pred[select_index]
                cur_bbox_pred = cur_bbox_pred[select_index]
                cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[select_index]

                # for rotation angle pred: [-pi, 0]
                cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(),
                                                    1,
                                                    (2 * np.pi / self.num_dir_bins)).to(cur_bbox_pred.device)
                cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * (2 * np.pi / self.num_dir_bins)

                bbox_pred_final.append(cur_bbox_pred)
                bbox_pred_label_final.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + j)
                bbox_cls_score_final.append(cur_bbox_cls_pred)

            # Filter bboxes if their number is above self.nms_post_maxsize
            if len(bbox_pred_final) > 0:
                bbox_pred_final = torch.cat(bbox_pred_final, 0)
                bbox_pred_label_final = torch.cat(bbox_pred_label_final, 0)
                bbox_cls_score_final = torch.cat(bbox_cls_score_final, 0)
                if bbox_pred_final.size(0) > self.nms_post_maxsize:
                    final_inds = bbox_cls_score_final.topk(self.nms_post_maxsize)[1]
                    bbox_pred_final = bbox_pred_final[final_inds]
                    bbox_pred_label_final = bbox_pred_label_final[final_inds]
                    bbox_cls_score_final = bbox_cls_score_final[final_inds]
                pred_res = {
                    "bboxes3D": bbox_pred_final.detach().cpu().numpy(),
                    "labels": bbox_pred_label_final.detach().cpu().numpy(),
                    "scores": bbox_cls_score_final.detach().cpu().numpy()
                }
                batch_pred_bboxes.append(pred_res)
        return batch_pred_bboxes

    @staticmethod
    def convert_anchors2bboxes(anchors, bbox_pred):
        diagnal_xy_size = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
        x = bbox_pred[:, 0] * diagnal_xy_size + anchors[:, 0]
        y = bbox_pred[:, 1] * diagnal_xy_size + anchors[:, 1]
        z = bbox_pred[:, 2] * anchors[:, 5] + anchors[:, 2] + anchors[:, 5] / 2

        w = anchors[:, 3] * torch.exp(bbox_pred[:, 3])
        l = anchors[:, 4] * torch.exp(bbox_pred[:, 4])
        h = anchors[:, 5] * torch.exp(bbox_pred[:, 5])
        z = z - h / 2
        angle = anchors[:, 6] + bbox_pred[:, 6]

        bboxes = torch.stack([x, y, z, w, l, h, angle], dim=1)
        return bboxes

    def generate_batch_anchor_target(self, batched_anchors, batched_gt_bboxes, batched_gt_labels):
        """
        Use the anchors to generate the ground truth 3D boundary boxes targets for predictions
        :param batched_anchors: 3D prior anchor boxes
        :param batched_gt_bboxes: ground truth 3D bounding boxes
        :param batched_gt_labels: Correct object class labels for ground truth 3D bounding boxes
        :return: batch_anchor_target_dict contains the ground truth targets with respect to anchors for the training
        """
        assert len(batched_gt_bboxes) == len(batched_gt_labels) == len(batched_anchors)
        batch_size = len(batched_gt_bboxes)
        batched_labels, batched_label_weights = [], []
        batched_bbox_reg, batched_bbox_reg_weights = [], []
        batched_dir_labels, batched_dir_labels_weights = [], []
        for i in range(batch_size):
            anchors = batched_anchors[i]
            x_dim, y_dim, num_object_dim, rotation_dim, box_representation_dim = anchors.size()
            gt_bboxes, gt_labels = batched_gt_bboxes[i], batched_gt_labels[i]
            labels, label_weights = [], []
            bbox_reg, bbox_reg_weights = [], []
            dir_labels, dir_labels_weights = [], []
            for j in range(len(self.anchor_boxes_cfg)):
                cur_cls_cfg = self.anchor_boxes_cfg[j]
                pos_iou_thr = cur_cls_cfg.matched_threshold
                neg_iou_thr = cur_cls_cfg.unmatched_threshold
                min_iou_thr = cur_cls_cfg.unmatched_threshold

                cur_anchors = anchors[:, :, j, :, :].reshape(-1, 7)
                overlaps = iou2d(gt_bboxes, cur_anchors)
                max_overlaps, max_overlaps_idx = torch.max(overlaps, dim=0)  # for anchors
                gt_max_overlaps, _ = torch.max(overlaps, dim=1)  # for gt_bboxes

                assigned_gt_index = -1 * torch.ones_like(cur_anchors[:, 0], dtype=torch.long)
                assigned_gt_index[max_overlaps < neg_iou_thr] = 0  # for negative anchors
                assigned_gt_index[max_overlaps >= pos_iou_thr] = (
                        max_overlaps_idx[max_overlaps >= pos_iou_thr] + 1)  # for positive anchors
                # for anchors which are the highest iou
                for k in range(len(gt_bboxes)):
                    if gt_max_overlaps[k] >= min_iou_thr:
                        assigned_gt_index[overlaps[k] == gt_max_overlaps[k]] = k + 1

                pos_flag = assigned_gt_index > 0
                neg_flag = assigned_gt_index == 0
                # 1. anchor labels
                # negative samples are assigned to class self.num_object_classes
                assigned_gt_labels = torch.zeros_like(cur_anchors[:, 0],
                                                      dtype=torch.long) + self.num_object_classes
                assigned_gt_labels[pos_flag] = gt_labels[assigned_gt_index[pos_flag] - 1].long()
                assigned_gt_labels_weights = torch.zeros_like(cur_anchors[:, 0])
                assigned_gt_labels_weights[pos_flag] = 1
                assigned_gt_labels_weights[neg_flag] = 1

                # 2. anchor regression
                assigned_gt_reg_weights = torch.zeros_like(cur_anchors[:, 0])
                assigned_gt_reg_weights[pos_flag] = 1

                assigned_gt_reg = torch.zeros_like(cur_anchors)
                positive_anchors = cur_anchors[pos_flag]
                corr_gt_bboxes = gt_bboxes[assigned_gt_index[pos_flag] - 1]
                assigned_gt_reg[pos_flag] = self.convert_bboxes2loss_terms(corr_gt_bboxes, positive_anchors)

                # 3. anchor direction
                assigned_gt_dir_weights = torch.zeros_like(cur_anchors[:, 0])
                assigned_gt_dir_weights[pos_flag] = 1

                assigned_gt_dir = torch.zeros_like(cur_anchors[:, 0], dtype=torch.long)
                dir_cls_targets = limit_period(corr_gt_bboxes[:, 6].cpu(), 0, 2 * np.pi).to(corr_gt_bboxes.device)
                dir_cls_targets = torch.floor(dir_cls_targets / np.pi).long()
                assigned_gt_dir[pos_flag] = torch.clamp(dir_cls_targets, min=0, max=1)

                labels.append(assigned_gt_labels.reshape(x_dim, y_dim, 1, rotation_dim))
                label_weights.append(assigned_gt_labels_weights.reshape(x_dim, y_dim, 1, rotation_dim))
                bbox_reg.append(assigned_gt_reg.reshape(x_dim, y_dim, 1, rotation_dim, -1))
                bbox_reg_weights.append(assigned_gt_reg_weights.reshape(x_dim, y_dim, 1, rotation_dim))
                dir_labels.append(assigned_gt_dir.reshape(x_dim, y_dim, 1, rotation_dim))
                dir_labels_weights.append(assigned_gt_dir_weights.reshape(x_dim, y_dim, 1, rotation_dim))

            labels = torch.cat(labels, dim=-2).reshape(-1)
            label_weights = torch.cat(label_weights, dim=-2).reshape(-1)
            bbox_reg = torch.cat(bbox_reg, dim=-3).reshape(-1, box_representation_dim)
            bbox_reg_weights = torch.cat(bbox_reg_weights, dim=-2).reshape(-1)
            dir_labels = torch.cat(dir_labels, dim=-2).reshape(-1)
            dir_labels_weights = torch.cat(dir_labels_weights, dim=-2).reshape(-1)

            batched_labels.append(labels)
            batched_label_weights.append(label_weights)
            batched_bbox_reg.append(bbox_reg)
            batched_bbox_reg_weights.append(bbox_reg_weights)
            batched_dir_labels.append(dir_labels)
            batched_dir_labels_weights.append(dir_labels_weights)

        batch_anchor_target_dict = dict(
            batched_labels=torch.stack(batched_labels, 0),  # (bs, y_dim * x_dim * 3 * 2)
            batched_label_weights=torch.stack(batched_label_weights, 0),  # (bs, y_dim * x_dim * 3 * 2)
            batched_bbox_reg=torch.stack(batched_bbox_reg, 0),  # (bs, y_dim * x_dim * 3 * 2, 7)
            batched_bbox_reg_weights=torch.stack(batched_bbox_reg_weights, 0),  # (bs, y_dim * x_dim * 3 * 2)
            batched_dir_labels=torch.stack(batched_dir_labels, 0),  # (bs, y_dim * x_dim * 3 * 2)
            batched_dir_labels_weights=torch.stack(batched_dir_labels_weights, 0)  # (bs, y_dim * x_dim * 3 * 2)
        )
        return batch_anchor_target_dict

    @staticmethod
    def convert_bboxes2loss_terms(gt_bboxes, anchors):
        diagnal_xy_size = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
        dx = (gt_bboxes[:, 0] - anchors[:, 0]) / diagnal_xy_size
        dy = (gt_bboxes[:, 1] - anchors[:, 1]) / diagnal_xy_size

        zb = gt_bboxes[:, 2] + gt_bboxes[:, 5] / 2
        za = anchors[:, 2] + anchors[:, 5] / 2
        dz = (zb - za) / anchors[:, 5]

        dw = torch.log(gt_bboxes[:, 3] / anchors[:, 3])
        dl = torch.log(gt_bboxes[:, 4] / anchors[:, 4])
        dh = torch.log(gt_bboxes[:, 5] / anchors[:, 5])
        dtheta = gt_bboxes[:, 6] - anchors[:, 6]

        deltas = torch.stack([dx, dy, dz, dw, dl, dh, dtheta], dim=1)
        return deltas
    
    def generate_predictions_for_train(self, batched_data_dict, batched_anchors,
                                       bbox_cls_pred, bbox_pred, bbox_dir_cls_pred):
        batched_gt_bboxes = batched_data_dict["batched_gt_bboxes"]
        batched_gt_labels = batched_data_dict["batched_labels"]
        anchor_target_dict = self.generate_batch_anchor_target(batched_anchors,
                                                               batched_gt_bboxes,
                                                               batched_gt_labels)

        bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, self.num_object_classes)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

        batched_bbox_labels = anchor_target_dict["batched_labels"].reshape(-1)
        batched_label_weights = anchor_target_dict["batched_label_weights"].reshape(-1)
        batched_bbox_reg = anchor_target_dict["batched_bbox_reg"].reshape(-1, 7)
        batched_dir_labels = anchor_target_dict["batched_dir_labels"].reshape(-1)

        pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < self.num_object_classes)
        bbox_pred = bbox_pred[pos_idx]
        batched_bbox_reg = batched_bbox_reg[pos_idx]
        # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
        bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
        batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
        bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
        batched_dir_labels = batched_dir_labels[pos_idx]

        num_cls_pos = (batched_bbox_labels < self.num_object_classes).sum()
        bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
        batched_bbox_labels[batched_bbox_labels < 0] = self.num_object_classes
        batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]
        res_dict = EasyDict(
            dict(bbox_cls_pred=bbox_cls_pred,
                 bbox_pred=bbox_pred,
                 bbox_dir_cls_pred=bbox_dir_cls_pred,
                 batched_labels=batched_bbox_labels,
                 num_cls_pos=num_cls_pos,
                 batched_bbox_reg=batched_bbox_reg,
                 batched_dir_labels=batched_dir_labels))
        return res_dict