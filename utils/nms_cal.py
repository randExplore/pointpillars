import numpy as np
import cv2


# calculate nms for the selected bboxes index based on numpy

def nms_3d(boxes, scores, nms_thres=0.1, score_thres=0.6):
    # returns indices of selected bboxes

    def calculate_overlap(keep_box, rest_boxes, min_index, max_index):
        keep_box_min = keep_box[min_index] - keep_box[max_index] / 2
        keep_box_max = keep_box[min_index] + keep_box[max_index] / 2
        rest_boxes_min = rest_boxes[:, min_index] - rest_boxes[:, max_index] / 2
        rest_boxes_max = rest_boxes[:, min_index] + rest_boxes[:, max_index] / 2
        min_axis = np.maximum(rest_boxes_min, keep_box_min)
        max_axis = np.minimum(rest_boxes_max, keep_box_max)
        overlap = max_axis > min_axis
        return overlap

    sorted_index = np.argsort(-scores)
    keep_index = []
    while len(sorted_index) > 0:
        if scores[sorted_index[0]] < score_thres:
            break
        keep_index.append(sorted_index[0])
        if len(sorted_index) == 1:
            break
        keep_box = boxes[sorted_index[0], :]
        rest_index = sorted_index[1:]
        rest_boxes = boxes[rest_index, :]
        sel_inds = np.arange(len(rest_index))

        x_overlap = calculate_overlap(keep_box, rest_boxes, 0, 3)
        y_overlap = calculate_overlap(keep_box, rest_boxes, 1, 4)
        z_overlap = calculate_overlap(keep_box, rest_boxes, 2, 5)

        overlap_mask = x_overlap & y_overlap & z_overlap
        care_res_boxes = rest_boxes[overlap_mask, :]
        overlap_sel_inds = sel_inds[overlap_mask]
        iou3d_res = iou3d(keep_box, care_res_boxes)
        delete_mask = iou3d_res > nms_thres
        delete_sel_inds = overlap_sel_inds[delete_mask]
        sorted_index = np.delete(rest_index, delete_sel_inds)

    if len(keep_index) > 0:
        keep_index = np.array(keep_index)
        return keep_index
    else:
        return None


def iou3d(keep_box, rest_boxes, scale_factor=100):
    # calculate iou for 3d bboxes
    keep_box = keep_box.copy()
    rest_boxes = rest_boxes.copy()
    keep_box[: 6] *= scale_factor
    rest_boxes[:, :6] *= scale_factor
    all_boxes = np.concatenate((np.expand_dims(keep_box, axis=0), rest_boxes), axis=0)
    all_corners = boxes_to_corners_3d(all_boxes)  # the corner locations follow the definition inside boxes_to_corners_3d
    all_min_x = np.min(all_corners[:, :, 0])
    all_max_x = np.max(all_corners[:, :, 0])
    all_min_y = np.min(all_corners[:, :, 1])
    all_max_y = np.max(all_corners[:, :, 1])
    H = int(all_max_y - all_min_y)
    W = int(all_max_x - all_min_x)

    all_corners[:, :, :2] -= [all_min_x, all_min_y]
    keep_corners = all_corners[0, :, :]  # the bbox that is trying to keep
    res_corners = all_corners[1:, :, :]  # the remaining/rest bboxes

    blank_map = np.zeros((H, W))

    keep_mask = draw_mask(keep_corners, blank_map)

    # z will be handled here, xy will be hnadled using the image contour
    keep_box_min_z = keep_box[2] - keep_box[5] / 2
    keep_box_max_z = keep_box[2] + keep_box[5] / 2
    rest_boxes_min_z = rest_boxes[:, 2] - rest_boxes[:, 5] / 2
    rest_boxes_max_z = rest_boxes[:, 2] + rest_boxes[:, 5] / 2
    min_z = np.maximum(rest_boxes_min_z, keep_box_min_z)
    max_z = np.minimum(rest_boxes_max_z, keep_box_max_z)
    z_overlap = max_z - min_z  # (n, )

    ious = np.zeros(len(z_overlap))  # (n, )
    for i in range(len(z_overlap)):
        h_i = z_overlap[i]
        mask_i = draw_mask(res_corners[i, :, :], blank_map)
        overlap_i = h_i * (np.sum(mask_i & keep_mask))
        union = np.sum(keep_mask) * keep_box[5] + np.sum(mask_i) * rest_boxes[i, 5] - overlap_i
        ious[i] = overlap_i / union

    return ious


def draw_mask(corners, blank_map):
    # this function is used to calculate the xy area
    contours = corners[:4, :2]  # the corner locations follow the definition inside boxes_to_corners_3d
    contours = contours.astype(np.int32)
    H, W = blank_map.shape
    contours[:, 0] = np.clip(contours[:, 0], 0, W)
    contours[:, 1] = np.clip(contours[:, 1], 0, H)
    contours = np.expand_dims(contours, axis=1)  # (4, 1, 2), float
    img = blank_map.copy()

    cv2.drawContours(img, [contours], -1, 1, -1)
    return img.astype(np.bool_)   # 1 means the pixel is taken


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading] in lidar reference, (x, y, z) is the bbox center
    """

    template = np.array(
        [[1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]]) / 2

    corners3d = np.tile(boxes3d[:, None, 3:6], (1, 8, 1)) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3), boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def rotate_points_along_z(points, angle):
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0], dtype=points.dtype)
    ones = np.ones(points.shape[0], dtype=angle.dtype)
    rot_matrix = np.stack(
        (cosa, sina, zeros,
         -sina, cosa, zeros,
        zeros, zeros, ones), axis=1, dtype=np.float32).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def boxes_camera_to_corners_3d(boxes3d):
    """
         z
        /
       /
      /
    |o ------> x(right)
    |
    |
    |
    v y(down)

        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/    o    |/
      2 -------- 1
    boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading] in camera reference, (x, y, z) is the bbox center
    """
    template = np.array(
        [[1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 0, 1],
        [1, -2, 1], [1, -2, -1], [-1, -2, -1], [-1, -2, 1]]) / 2

    corners3d = np.tile(boxes3d[:, None, 3:6], (1, 8, 1)) * template[None, :, :]
    corners3d = rotate_points_along_y(corners3d.reshape(-1, 8, 3), boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def rotate_points_along_y(points, angle):
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0], dtype=points.dtype)
    ones = np.ones(points.shape[0], dtype=angle.dtype)
    rot_matrix = np.stack(
        (cosa, zeros, sina,
        zeros, ones, zeros,
        -sina, zeros, cosa), axis=1, dtype=np.float32).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot
