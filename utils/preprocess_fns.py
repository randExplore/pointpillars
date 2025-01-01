import numpy as np
import numba

# copied and modified from: https://github.com/zhulf0804/PointPillars/blob/main/utils/process.py

# These functions are used for preprocessing data
def projection_matrix_to_CRT_kitti(proj):
    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T


def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array(
        [near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array(
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]],
        dtype=C.dtype)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners],
                            axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz


def points_camera2lidar(points, tr_velo_to_cam, r0_rect):
    extended_xyz = np.pad(points, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    return xyz[..., :3]


def group_rectangle_vertexs(bboxes_corners):
    rec1 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 1], bboxes_corners[:, 3], bboxes_corners[:, 2]],
                    axis=1)  # (n, 4, 3)
    rec2 = np.stack([bboxes_corners[:, 4], bboxes_corners[:, 7], bboxes_corners[:, 6], bboxes_corners[:, 5]],
                    axis=1)  # (n, 4, 3)
    rec3 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 4], bboxes_corners[:, 5], bboxes_corners[:, 1]],
                    axis=1)  # (n, 4, 3)
    rec4 = np.stack([bboxes_corners[:, 2], bboxes_corners[:, 6], bboxes_corners[:, 7], bboxes_corners[:, 3]],
                    axis=1)  # (n, 4, 3)
    rec5 = np.stack([bboxes_corners[:, 1], bboxes_corners[:, 5], bboxes_corners[:, 6], bboxes_corners[:, 2]],
                    axis=1)  # (n, 4, 3)
    rec6 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 3], bboxes_corners[:, 7], bboxes_corners[:, 4]],
                    axis=1)  # (n, 4, 3)
    group_rectangle_vertexs = np.stack([rec1, rec2, rec3, rec4, rec5, rec6], axis=1)
    return group_rectangle_vertexs


def group_plane_equation(bbox_group_rectangle_vertexs):
    vectors = bbox_group_rectangle_vertexs[:, :, :2] - bbox_group_rectangle_vertexs[:, :, 1:3]
    normal_vectors = np.cross(vectors[:, :, 0], vectors[:, :, 1])  # (n, 6, 3)
    normal_d = np.einsum('ijk,ijk->ij', bbox_group_rectangle_vertexs[:, :, 0], normal_vectors)  # (n, 6)
    plane_equation_params = np.concatenate([normal_vectors, -normal_d[:, :, None]], axis=-1)
    return plane_equation_params


@numba.jit(nopython=True)
def points_in_bboxes(points, plane_equation_params):
    '''
    points: shape=(N, 3)
    plane_equation_params: shape=(n, 6, 4)
    return: shape=(N, n), bool
    '''
    N, n = len(points), len(plane_equation_params)
    m = plane_equation_params.shape[1]
    masks = np.ones((N, n), dtype=np.bool_)
    for i in range(N):
        x, y, z = points[i, :3]
        for j in range(n):
            bbox_plane_equation_params = plane_equation_params[j]
            for k in range(m):
                a, b, c, d = bbox_plane_equation_params[k]
                if a * x + b * y + c * z + d >= 0:
                    masks[i][j] = False
                    break
    return masks


def get_points_in_bboxes(points, r0_rect, tr_velo_to_cam, dimensions, location, rotation_y, name):
    n_total_bbox = len(dimensions)
    n_valid_bbox = len([item for item in name if item != 'DontCare'])
    location, dimensions = location[:n_valid_bbox], dimensions[:n_valid_bbox]
    rotation_y, name = rotation_y[:n_valid_bbox], name[:n_valid_bbox]
    bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=1)
    bboxes_lidar = bbox_camera2lidar(bboxes_camera, tr_velo_to_cam, r0_rect)
    bboxes_corners = bbox3d2corners(bboxes_lidar)
    group_rectangle_vertexs_v = group_rectangle_vertexs(bboxes_corners)
    frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
    num_index = points_in_bboxes(points[:, :3], frustum_surfaces)  # (N, n), N is points num, n is bboxes number
    return num_index, n_total_bbox, n_valid_bbox, bboxes_lidar, name

def get_points_num_in_bbox(points, r0_rect, tr_velo_to_cam, dimensions, location, rotation_y, name):
    indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
        get_points_in_bboxes(
            points=points,
            r0_rect=r0_rect,
            tr_velo_to_cam=tr_velo_to_cam,
            dimensions=dimensions,
            location=location,
            rotation_y=rotation_y,
            name=name)
    points_num = np.sum(indices, axis=0)
    non_valid_points_num = [-1] * (n_total_bbox - n_valid_bbox)
    points_num = np.concatenate([points_num, non_valid_points_num], axis=0)
    return np.array(points_num, dtype=np.int32)


def bbox_camera2lidar(bboxes, tr_velo_to_cam, r0_rect):
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([z_size, x_size, y_size], axis=1)
    extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    bboxes_lidar = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return np.array(bboxes_lidar, dtype=np.float32)


def bbox3d2corners(bboxes):
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]
    bboxes_corners = np.array([[-0.5, -0.5, 0], [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [-0.5, 0.5, 0.0],
                               [0.5, -0.5, 0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [0.5, 0.5, 0.0]],
                              dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :]  # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    rot_sin, rot_cos = np.sin(angles), np.cos(angles)

    rot_mat = np.array([[rot_cos, rot_sin, np.zeros_like(rot_cos)],
                        [-rot_sin, rot_cos, np.zeros_like(rot_cos)],
                        [np.zeros_like(rot_cos), np.zeros_like(rot_cos), np.ones_like(rot_cos)]],
                       dtype=np.float32)  # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0))  # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat  # (n, 8, 3)

    bboxes_corners += centers[:, None, :]
    return bboxes_corners


def bbox3d2corners_camera(bboxes):
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]

    bboxes_corners = np.array([[0.5, 0.0, -0.5], [0.5, -1.0, -0.5], [-0.5, -1.0, -0.5], [-0.5, 0.0, -0.5],
                               [0.5, 0.0, 0.5], [0.5, -1.0, 0.5], [-0.5, -1.0, 0.5], [-0.5, 0.0, 0.5]],
                              dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :]  # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    rot_sin, rot_cos = np.sin(angles), np.cos(angles)

    rot_mat = np.array([[rot_cos, np.zeros_like(rot_cos), rot_sin],
                        [np.zeros_like(rot_cos), np.ones_like(rot_cos), np.zeros_like(rot_cos)],
                        [-rot_sin, np.zeros_like(rot_cos), rot_cos]],
                       dtype=np.float32)  # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0))  # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat  # (n, 8, 3)

    bboxes_corners += centers[:, None, :]
    return bboxes_corners


# The following functions are used for data augmentation purpose and 3d_bboxes conversions
def remove_pts_in_bboxes(points, bboxes, rm=True):
    bboxes_corners = bbox3d2corners(bboxes)  # (n, 8, 3)
    bbox_group_rectangle_vertexs = group_rectangle_vertexs(bboxes_corners)  # (n, 6, 4, 3)

    group_plane_equation_params = group_plane_equation(bbox_group_rectangle_vertexs)

    masks = points_in_bboxes(points, group_plane_equation_params)  # (N, n)
    if rm:
        masks = np.any(masks, axis=-1)
        return points[~masks]
    else:
        return masks


def limit_period(val, offset=0.5, period=np.pi):
    # value in [-offset * period, (1-offset) * period]
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val


def bbox3d2bevcorners(bboxes):
    centers, dims, angles = bboxes[:, :2], bboxes[:, 3:5], bboxes[:, 6]
    bev_corners = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=np.float32)
    bev_corners = bev_corners[None, ...] * dims[:, None, :]  # (1, 4, 2) * (n, 1, 2) -> (n, 4, 2)

    rot_sin, rot_cos = np.sin(angles), np.cos(angles)

    rot_mat = np.array([[rot_cos, rot_sin],
                        [-rot_sin, rot_cos]])  # (2, 2, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0))  # (N, 2, 2)
    bev_corners = bev_corners @ rot_mat  # (n, 4, 2)

    bev_corners += centers[:, None, :]
    return bev_corners.astype(np.float32)


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]),
                           axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = bevcorner2alignedbbox(boxes)
    qboxes_standup = bevcorner2alignedbbox(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (
                    min(boxes_standup[i, 2], qboxes_standup[j, 2]) -
                    max(boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (
                        min(boxes_standup[i, 3], qboxes_standup[j, 3]) -
                        max(boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for box_l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, box_l, 0]
                            D = lines_qboxes[j, box_l, 1]
                            acd = (D[1] - A[1]) * (C[0] -
                                                   A[0]) > (C[1] - A[1]) * (
                                          D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] -
                                                   B[0]) > (C[1] - B[1]) * (
                                          D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                        B[1] - A[1]) * (
                                              C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                        B[1] - A[1]) * (
                                              D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for box_l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                        boxes[i, k, 0] - qboxes[j, box_l, 0])
                                cross -= vec[0] * (
                                        boxes[i, k, 1] - qboxes[j, box_l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for box_l in range(4):  # point box_l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                            qboxes[j, k, 0] - boxes[i, box_l, 0])
                                    cross -= vec[0] * (
                                            qboxes[j, k, 1] - boxes[i, box_l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret


@numba.jit(nopython=True)
def bevcorner2alignedbbox(bev_corners):
    n = len(bev_corners)
    alignedbbox = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        cur_bev = bev_corners[i]
        alignedbbox[i, 0] = np.min(cur_bev[:, 0])
        alignedbbox[i, 2] = np.max(cur_bev[:, 0])
        alignedbbox[i, 1] = np.min(cur_bev[:, 1])
        alignedbbox[i, 3] = np.max(cur_bev[:, 1])
    return alignedbbox