# -*- encoding: utf-8 -*-
"""
@File    :   points_in_rbbox.py
@Time    :   2025/04/18 15:35:40
@Author  :   lh9171338
@Version :   1.0
@Contact :   2909171338@qq.com
"""

import numpy as np
import torch
from . import points_in_rbbox_ops


def points_in_rbbox_numpy(points, boxes, return_indices=False):
    """
    find points in rbbox with numpy

    Args:
        points (np.ndarray): points with shape [N, 3]
        boxes (np.ndarray): boxes with shape [M, 7]
        return_indices (bool): whether to return indices, default is False

    Returns:
        mask (np.ndarray): mask with shape [M, N], only return when `return_indices` is False
        indices_list (list[np.ndarray]): indices list with length M, only return when `return_indices` is True
    """
    N = points.shape[0]
    M = boxes.shape[0]
    if N == 0 or M == 0:
        if return_indices:
            return [[]] * M
        else:
            return np.zeros((M, N), dtype="bool")

    centers = boxes[:, :3]
    dim = boxes[:, 3:6]
    theta = boxes[:, 6]
    points_local = points[None, :, :3] - centers[:, None]  # shape [M, N, 3]
    rot_matrix = np.zeros((M, 3, 3))  # shape [M, 3, 3]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot_matrix[:, 0, 0] = cos_theta
    rot_matrix[:, 0, 1] = -sin_theta
    rot_matrix[:, 1, 0] = sin_theta
    rot_matrix[:, 1, 1] = cos_theta
    rot_matrix[:, 2, 2] = 1.0

    points_local = points_local @ rot_matrix  # shape [M, N, 3]
    points_dist = np.abs(points_local)
    mask = (points_dist < 0.5 * dim[:, None]).all(axis=-1)
    if return_indices:
        indices = np.flatnonzero(mask)
        indices %= N
        indices_list = np.split(indices, mask.sum(axis=-1).cumsum()[:-1])
        return indices_list
    else:
        return mask


@torch.no_grad()
def points_in_rbbox_torch(points, boxes, device="cuda", dtype=torch.float32, return_indices=False):
    """
    find points in rbbox with torch

    Args:
        points (np.ndarray|torch.Tensor): points with shape [N, 3]
        boxes (np.ndarray|torch.Tensor): boxes with shape [M, 7]
        device (str|int): device, default is 'cuda'
        dtype (torch.dtype): data type, default is torch.float32, choose from [torch.float32, torch.float16, torch.bfloat16]
        return_indices (bool): whether to return indices, default is False

    Returns:
        mask (np.ndarray): mask with shape [M, N], only return when `return_indices` is False
        indices_list (list[np.ndarray]): indices list with length M, only return when `return_indices` is True
    """
    N = points.shape[0]
    M = boxes.shape[0]
    if N == 0 or M == 0:
        if return_indices:
            return [[]] * M
        else:
            return np.zeros((M, N), dtype="bool")

    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
    points = points.to(device=device, dtype=dtype)
    boxes = boxes.to(device=device, dtype=dtype)

    centers = boxes[:, :3]
    dim = boxes[:, 3:6]
    theta = boxes[:, 6]
    points_local = points[None, :, :3] - centers[:, None]  # shape [M, N, 3]

    rot_matrix = points.new_zeros((M, 3, 3))  # shape [M, 3, 3]
    cos_theta = theta.cos()
    sin_theta = theta.sin()
    rot_matrix[:, 0, 0] = cos_theta
    rot_matrix[:, 0, 1] = -sin_theta
    rot_matrix[:, 1, 0] = sin_theta
    rot_matrix[:, 1, 1] = cos_theta
    rot_matrix[:, 2, 2] = 1.0

    points_local = points_local @ rot_matrix  # shape [M, N, 3]
    points_local = points_local.abs()
    mask = (points_local < 0.5 * dim[:, None]).all(dim=-1)
    if return_indices:
        indices = mask.flatten().nonzero(as_tuple=True)[0]
        indices %= N
        indices_list = indices.split(mask.sum(dim=-1).tolist())
        indices_list = [x.cpu().numpy() for x in indices_list]
        return indices_list
    else:
        return mask.cpu().numpy()


@torch.no_grad()
def points_in_rbbox_cuda(points, boxes, device="cuda", dtype=torch.float32, return_indices=False):
    """
    find points in rbbox with torch

    Args:
        points (np.ndarray|torch.Tensor): points with shape [N, 3]
        boxes (np.ndarray|torch.Tensor): boxes with shape [M, 7]
        device (str|int): device, default is 'cuda'
        dtype (torch.dtype): data type, default is torch.float32, choose from [torch.float32, torch.float16, torch.bfloat16]
        return_indices (bool): whether to return indices, default is False

    Returns:
        mask (np.ndarray): mask with shape [M, N], only return when `return_indices` is False
        indices_list (list[np.ndarray]): indices list with length M, only return when `return_indices` is True
    """
    if device == "cpu":
        return points_in_rbbox_torch(points, boxes, device=device, dtype=dtype, return_indices=return_indices)

    N = points.shape[0]
    M = boxes.shape[0]
    if N == 0 or M == 0:
        if return_indices:
            return [[]] * M
        else:
            return np.zeros((M, N), dtype="bool")

    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
    points = points.to(device=device, dtype=dtype)
    boxes = boxes.to(device=device, dtype=dtype)
    mask = points.new_zeros((M, N), dtype=torch.bool)
    points_in_rbbox_ops.points_in_rbbox_wrapper(points, boxes, mask)
    if return_indices:
        indices = mask.flatten().nonzero(as_tuple=True)[0]
        indices %= N
        indices_list = indices.split(mask.sum(dim=-1).tolist())
        indices_list = [x.cpu().numpy() for x in indices_list]
        return indices_list
    else:
        return mask.cpu().numpy()
