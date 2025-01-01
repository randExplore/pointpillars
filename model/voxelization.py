import numpy as np
import numba


def voxelize(point_cloud_data, voxel_size, coordinates_range, max_num_points=64, max_num_voxels=20000):
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=point_cloud_data.dtype)
    if not isinstance(coordinates_range, np.ndarray):
        coordinates_range = np.array(coordinates_range, dtype=point_cloud_data.dtype)
    grid_size = (coordinates_range[3:] - coordinates_range[:3]) / voxel_size  # [x, y, z]
    grid_size = tuple(np.round(grid_size).astype(np.int32).tolist())

    num_points_per_voxel = np.zeros(shape=(max_num_voxels,), dtype=np.int32)
    coordinates2voxel_idx = -np.ones(shape=grid_size, dtype=np.int32)
    voxels = np.zeros(shape=(max_num_voxels, max_num_points, point_cloud_data.shape[-1]), dtype=point_cloud_data.dtype)
    coordinates = np.zeros(shape=(max_num_voxels, 3), dtype=np.int32)
    voxel_num = voxel_core(point_cloud_data, voxel_size, coordinates_range, grid_size, num_points_per_voxel,
                           coordinates2voxel_idx, voxels, coordinates, max_num_points, max_num_voxels)
    voxels, coordinates, num_points_per_voxel = (voxels[:voxel_num],
                                                 coordinates[:voxel_num], num_points_per_voxel[:voxel_num])
    return voxels, coordinates, num_points_per_voxel


@numba.jit(nopython=True)
def voxel_core(point_cloud_data, voxel_size, coordinates_range, grid_size, num_points_per_voxel,
               coordinates2voxel_idx, voxels, coordinates, max_num_points=64, max_num_voxels=20000):
    n = point_cloud_data.shape[0]
    ndim = point_cloud_data.shape[1] - 1
    cur_coordinate = np.zeros(shape=(ndim,), dtype=np.int32)
    voxel_num = 0
    for i in range(n):
        failed = False
        for j in range(ndim):
            c = np.floor((point_cloud_data[i, j] - coordinates_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            cur_coordinate[j] = c
        if failed:
            continue
        voxel_idx = coordinates2voxel_idx[cur_coordinate[0], cur_coordinate[1], cur_coordinate[2]]
        if voxel_idx == -1:
            voxel_idx = voxel_num
            if voxel_num >= max_num_voxels:
                break
            voxel_num += 1
            coordinates2voxel_idx[cur_coordinate[0], cur_coordinate[1], cur_coordinate[2]] = voxel_idx
            coordinates[voxel_idx] = cur_coordinate
        num = num_points_per_voxel[voxel_idx]
        if num < max_num_points:
            voxels[voxel_idx, num] = point_cloud_data[i]
            num_points_per_voxel[voxel_idx] += 1
    return voxel_num
