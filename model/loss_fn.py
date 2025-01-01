import torch
import torch.nn as nn
import torch.nn.functional as F

# modified from: https://github.com/zhulf0804/PointPillars/blob/main/loss/loss.py


class Loss3DDetection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
        self.cls_w = cfg.cls_w
        self.reg_w = cfg.reg_w
        self.dir_w = cfg.dir_w

        # define the loss functions
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction="none",
                                              beta=cfg.beta)
        self.dir_cls = nn.CrossEntropyLoss()

    def forward(self, res_dict):
        bbox_cls_pred = res_dict.bbox_cls_pred  # [N, 3]
        bbox_pred = res_dict.bbox_pred  # [N, 7]
        bbox_dir_cls_pred = res_dict.bbox_dir_cls_pred  # [N, 2]
        batched_labels = res_dict.batched_labels  # [N, ]
        num_cls_pos = res_dict.num_cls_pos
        batched_bbox_reg = res_dict.batched_bbox_reg  # [N, 7]
        batched_dir_labels = res_dict.batched_dir_labels  # [N, ]

        # focal loss for bbox classification
        nclasses = bbox_cls_pred.size(1)
        batched_labels = F.one_hot(batched_labels, nclasses + 1)[:, :nclasses].float()
        bbox_cls_pred_sigmoid = torch.sigmoid(bbox_cls_pred)
        weights = self.alpha * (1 - bbox_cls_pred_sigmoid).pow(self.gamma) * batched_labels + \
                  (1 - self.alpha) * bbox_cls_pred_sigmoid.pow(self.gamma) * (1 - batched_labels)
        cls_loss = F.binary_cross_entropy(bbox_cls_pred_sigmoid, batched_labels, reduction="none")
        cls_loss = cls_loss * weights
        cls_loss = cls_loss.sum() / num_cls_pos

        # regression loss
        reg_loss = self.smooth_l1_loss(bbox_pred, batched_bbox_reg)
        reg_loss = reg_loss.sum() / reg_loss.size(0)

        # direction cls loss
        dir_cls_loss = self.dir_cls(bbox_dir_cls_pred, batched_dir_labels)

        # total loss
        loss = self.cls_w * cls_loss + self.reg_w * reg_loss + self.dir_w * dir_cls_loss
        return loss