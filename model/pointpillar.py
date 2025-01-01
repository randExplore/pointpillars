from .pillar_feature_extraction_layer import PillarFeatureExtraction
from .backbone import BackBone, Neck
from .detection_head import DetectionHead
from torch import nn


class PointPillar(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pillar_encoder = PillarFeatureExtraction(cfg)
        self.backbone = BackBone(cfg)
        self.neck = Neck(cfg)
        self.head = DetectionHead(cfg)

    def forward(self, batched_data_dict):
        batched_pts= batched_data_dict["batched_pts"]
        batch_bev_pillar_features = self.pillar_encoder(batched_pts)
        backbone_features = self.backbone(batch_bev_pillar_features)
        neck_features = self.neck(backbone_features)
        result = self.head(neck_features, len(batched_pts), batched_data_dict)
        return result
