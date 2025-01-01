from easydict import EasyDict

cfg = EasyDict({
    # data cfg
    "root_path": ".",
    "point_dim": 4,
    "class_name": EasyDict({
        "Pedestrian": 0,
        "Cyclist": 1,
        "Car": 2,
    }),

    "include_img_data": False,

    # model running mode
    "mode": "training",  # ["training", "testing"]

    # data split
    "split": "train",

    # device for training and testing
    "device": "gpu",  # using cpu or gpu

    # pillar cfg
    "pc_range": [0, -39.68, -3, 69.12, 39.68, 1],
    "voxel_size": [0.16, 0.16, 4],
    "max_num_points": 64,
    "max_voxels": [16000, 40000],

    # pillar feature extraction layer cfg
    "pfe_input_dim": 10,
    "bev_feature_dim": 64,

    # backbone cfg
    "conv_num_layers": [3, 5, 5],
    "conv_num_filters": [64, 128, 256],
    "conv_layer_strides": [2, 2, 2],
    "deConv_strides": [1, 2, 4],
    "deConv_num_filters": [128, 128, 128],

    # anchor_boxes cfg
    "anchor_boxes_cfg": [

        EasyDict({
            "class_name": "Pedestrian",
            "anchor_sizes": [[0.6, 0.8, 1.73]],
            "anchor_rotations": [0, 1.57],
            "anchor_bottom_heights": [-0.6],
            "matched_threshold": 0.5,
            "unmatched_threshold": 0.35
        }),
        EasyDict({
            "class_name": "Cyclist",
            "anchor_sizes": [[0.6, 1.76, 1.73]],
            "anchor_rotations": [0, 1.57],
            "anchor_bottom_heights": [-0.6],
            "matched_threshold": 0.5,
            "unmatched_threshold": 0.35
        }),
        EasyDict({
            "class_name": "Car",
            "anchor_sizes": [[1.6, 3.9, 1.56]],
            "anchor_rotations": [0, 1.57],
            "anchor_bottom_heights": [-1.78],
            "matched_threshold": 0.6,
            "unmatched_threshold": 0.45
        })
    ],

    # head cfg
    "input_channels": 384,
    "dir_offset": 0.78539,
    "dir_limit_offset": 0.0,
    "num_dir_bins": 2,

    # nms cfg
    "score_thr": 0.1,
    "nms_cfg": EasyDict({
        "nms_th": 0.01,
        "nms_pre_maxsize": 100,  # 4096
        "nms_post_maxsize": 50  # 500
    }),

    # loss cfg
    "alpha": 0.25,
    "gamma": 2.0,
    "beta": 1 / 9,
    "cls_w": 1.0,
    "reg_w": 2.0,
    "dir_w": 0.2,

    # train and test cfg
    "seed": 0,
    "data_augmentaion": True,
    "batch_size": 6,
    "max_epoch": 160,
    "init_lr": 2.5e-4,
    "log_freq": 100,
    "ckpt_freq_epoch": 10,
})
