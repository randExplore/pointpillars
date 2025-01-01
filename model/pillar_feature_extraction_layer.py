import torch
from torch import nn
import torch.nn.functional as F
from .voxelization import voxelize

# modified and referenced from:
# https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/data_preprocessors/voxelize.py,
# https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/voxel_encoders/pillar_encoder.py,
# https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_3d/vfe/pillar_vfe.py
# https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/middle_encoders/pillar_scatter.py
# https://github.com/zhulf0804/PointPillars/blob/main/model/pointpillars.py

class PillarLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        (voxel_size, point_cloud_range,
         max_num_points, max_voxels) = (cfg.voxel_size, cfg.pc_range,
                                        cfg.max_num_points, cfg.max_voxels)
        if self.training:
            max_voxels = max_voxels[0]
        else:
            max_voxels = max_voxels[1]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels

    @torch.no_grad()
    def forward(self, batched_pts):
        pillars, coordinates, num_points_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            device = pts.device
            voxels_out, coors_out, num_points_per_voxel_out = voxelize(pts.detach().cpu().numpy(),
                                                                       self.voxel_size,
                                                                       self.point_cloud_range,
                                                                       self.max_num_points,
                                                                       self.max_voxels)
            pillars.append(torch.from_numpy(voxels_out).to(device))
            coordinates.append(torch.from_numpy(coors_out).long().to(device))
            num_points_per_pillar.append(torch.from_numpy(num_points_per_voxel_out).to(device))

        pillars = torch.cat(pillars, dim=0)
        num_points_per_pillar = torch.cat(num_points_per_pillar, dim=0)
        coordinates_batch = []
        for i, cur_coors in enumerate(coordinates):
            coordinates_batch.append(F.pad(cur_coors, (0, 1), value=i))  # add batch index to differentiate each batch
        coordinates_batch = torch.cat(coordinates_batch, dim=0)
        return pillars, coordinates_batch, num_points_per_pillar


class PillarFeatureExtraction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim, out_dim = cfg.pfe_input_dim, cfg.bev_feature_dim
        self.vx, self.vy = cfg.voxel_size[0], cfg.voxel_size[1]
        self.x_offset = self.vx / 2 + cfg.pc_range[0]
        self.y_offset = self.vy / 2 + cfg.pc_range[1]
        self.x_l = int((cfg.pc_range[3] - cfg.pc_range[0]) / self.vx)
        self.y_l = int((cfg.pc_range[4] - cfg.pc_range[1]) / self.vy)
        self.pc_range = cfg.pc_range

        # voxelization
        self.pillar_layer = PillarLayer(cfg)

        # feature extraction layer
        self.linear = nn.Linear(input_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, eps=1e-3, momentum=0.01)

        # scatter pillar layer
        self.scatter = ScatterPillar(cfg)

    def forward(self, batched_pts):
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        device = pillars.device

        # offset_pt_center shape: [pb1 + pb2 + ... + pbn, num_points, 3]
        offset_pt_center = (pillars[:, :, :3] -
                            torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:,
                                                                                                   None,
                                                                                                   None])

        x_offset_pi_center = pillars[:, :, :1] - (
                coors_batch[:, None, 0:1] * self.vx + self.x_offset)  # [pb1 + pb2 + ... + pbn, num_points, 1]
        y_offset_pi_center = pillars[:, :, 1:2] - (
                coors_batch[:, None, 1:2] * self.vy + self.y_offset)  # [pb1 + pb2 + ... + pbn, num_points, 1]

        z_offset_pi_center = torch.tensor([(self.pc_range[5] + self.pc_range[2]) / 2]).to(x_offset_pi_center)
        z_offset_pi_center = z_offset_pi_center.repeat(x_offset_pi_center.shape)

        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center,
                              z_offset_pi_center], dim=-1)  # [pb1 + pb2 + ... + pbn, num_points, 10]
        features[:, :, 0:1] = x_offset_pi_center  # tmp
        features[:, :, 1:2] = y_offset_pi_center  # tmp

        voxel_ids = torch.arange(0, pillars.size(1)).to(device)  # [num_points, ]
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]
        mask = mask.permute(1, 0).contiguous()  # [pb1 + pb2 + ... + pbn, num_points]
        features *= mask[:, :, None]

        x = self.linear(features) # [pb1 + pb2 + ... + pbn, num_points, 64]
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x_max = torch.max(x, dim=1)[0]

        batch_size = coors_batch[-1, -1] + 1  # batch index is added in the data loader for the scatter pillar use
        batch_bev_pillar_features = self.scatter(x_max, coors_batch, batch_size)
        return batch_bev_pillar_features


class ScatterPillar(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.voxel_size = cfg.voxel_size
        self.pc_range = cfg.pc_range
        self.bev_feature_dim = cfg.bev_feature_dim
        self.grid_x_num = int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0])
        self.grid_y_num = int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1])

    def forward(self, pillar_extract_feats, pillar_coords, batch_size):
        batch_bev_pillar_features = []
        for batch_idx in range(batch_size):
            scatter_pillar_feature = torch.zeros(
                (self.grid_x_num, self.grid_y_num, self.bev_feature_dim),
                dtype=pillar_extract_feats.dtype,
                device=pillar_extract_feats.device
            )

            # get point cloud data for the current batch index
            batch_mask = pillar_coords[:, -1] == batch_idx
            cur_coords = pillar_coords[batch_mask, :]
            cur_features = pillar_extract_feats[batch_mask, :]  # [num_points_in_this_lidar_cloud, self.num_bev_features]
            scatter_pillar_feature[cur_coords[:, 0], cur_coords[:, 1]] = cur_features
            scatter_pillar_feature = scatter_pillar_feature.permute(2, 1, 0).contiguous()
            batch_bev_pillar_features.append(scatter_pillar_feature)

        batch_bev_pillar_features = torch.stack(batch_bev_pillar_features, 0) # [batch, num_bev_features, grid_y_num, grid_x_num]
        return batch_bev_pillar_features
