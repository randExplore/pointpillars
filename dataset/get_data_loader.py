import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def collate_train_func(batch_info_list):
    batched_pts_list, batched_gt_bboxes_list, batched_gt_bboxes_camera_list = [], [], []
    batched_labels_list, batched_names_list = [], []
    batched_difficulty_list = []
    batched_img_info_list, batched_calib_list = [], []
    batched_image_list = []
    batched_gt_bbox2d_list = []
    for cur_data_dict in batch_info_list:
        pts, gt_bboxes_3d = cur_data_dict["pts"], cur_data_dict["gt_bboxes_3d"]
        gt_labels, gt_names = cur_data_dict["gt_labels"], cur_data_dict["gt_object_name"]
        difficulty = cur_data_dict["difficulty"]
        image_info, calbi_info = cur_data_dict["image_info"], cur_data_dict["calib_info"]
        gt_bbox2d, gt_bboxes_3d_camera = cur_data_dict["gt_bbox2d"], cur_data_dict["gt_bboxes_3d_camera"]

        batched_pts_list.append(torch.from_numpy(pts))
        batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d))
        batched_labels_list.append(torch.from_numpy(gt_labels))
        batched_names_list.append(gt_names)
        batched_difficulty_list.append(torch.from_numpy(difficulty))
        batched_img_info_list.append(image_info)
        batched_calib_list.append(calbi_info)
        batched_gt_bbox2d_list.append(torch.from_numpy(gt_bbox2d))
        batched_gt_bboxes_camera_list.append(gt_bboxes_3d_camera)


        if "image" in cur_data_dict:
            # Then the image transformations can be added here
            transform = transforms.ToTensor()
            img_tensor = transform(cur_data_dict["image"])
            batched_image_list.append(img_tensor)

    batched_data_dict = dict(
        batched_pts=batched_pts_list,
        batched_gt_bboxes=batched_gt_bboxes_list,
        batched_labels=batched_labels_list,
        batched_names=batched_names_list,
        batched_difficulty=batched_difficulty_list,
        batched_img_info=batched_img_info_list,
        batched_calib_info=batched_calib_list,
        batched_gt_bboxes_3d_camera=batched_gt_bboxes_camera_list,
        batched_gt_2dbboxes_list=batched_gt_bbox2d_list
    )
    if len(batched_image_list) > 0:
        # batched_images = torch.cat(batched_image_list, dim=0).float()
        batched_data_dict["batched_images"] = batched_image_list
    return batched_data_dict


def get_trainval_data_loader_fn(dataset, batch_size, num_workers=0, shuffle=True):
    # this contains ground truth labels
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_train_func,
    )
