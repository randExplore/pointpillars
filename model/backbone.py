import torch
from torch import nn

# modified and referenced from:
# https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/backbones/second.py
# https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/necks/second_fpn.py


class BackBone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channel = cfg.bev_feature_dim
        conv_num_layers = cfg.conv_num_layers
        conv_num_filters = cfg.conv_num_filters
        conv_layer_strides = cfg.conv_layer_strides

        assert len(conv_num_filters) == len(conv_num_layers)
        assert len(conv_num_filters) == len(conv_layer_strides)

        self.multi_blocks = nn.ModuleList()
        for i in range(len(conv_layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(in_channel, conv_num_filters[i], 3,
                                    stride=conv_layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(conv_num_filters[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(conv_num_layers[i]):
                blocks.append(nn.Conv2d(conv_num_filters[i], conv_num_filters[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(conv_num_filters[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = conv_num_filters[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


class Neck(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels, upsample_strides, deConv_num_filters = cfg.conv_num_filters, cfg.deConv_strides, cfg.deConv_num_filters
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(deConv_num_filters)

        self.deConv_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            deConv_block = []
            deConv_block.append(nn.ConvTranspose2d(in_channels[i],
                                                    deConv_num_filters[i],
                                                    upsample_strides[i],
                                                    stride=upsample_strides[i],
                                                    bias=False))
            deConv_block.append(nn.BatchNorm2d(deConv_num_filters[i], eps=1e-3, momentum=0.01))
            deConv_block.append(nn.ReLU(inplace=True))

            self.deConv_blocks.append(nn.Sequential(*deConv_block))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        outs = []
        for i in range(len(self.deConv_blocks)):
            xi = self.deConv_blocks[i](x[i])  # [batch_size, 128, 248, 216]
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out
