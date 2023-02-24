# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pdb

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Norm_Loss(nn.Module):
    def __init__(self, vec_length_list, eps=1e-6, normalize=False, diff_order='L1'):
        super().__init__()
        self.vec_length_list = vec_length_list
        self.eps = eps
        self.normalize = normalize
        self.diff_order = diff_order

    def normal(self, pose, vec_length):
        # get surface
        start = pose[:, :, :-vec_length, :]
        end = pose[:, :, vec_length:, :]

        x = start[:, 1, :, :] * end[:, 2, :, :] - \
            start[:, 2, :, :] * end[:, 1, :, :]

        y = \
            start[:, 2, :, :] * end[:, 0, :, :] - \
            start[:, 0, :, :] * end[:, 2, :, :]

        z = \
            start[:, 0, :, :] * end[:, 1, :, :] - \
            start[:, 1, :, :] * end[:, 0, :, :]

        # pred_3d_norm = torch.cat([pred_3d_x, pred_3d_y, pred_3d_z], dim=2)
        # pred_3d_norm = torch.cat([pred_3d_x, pred_3d_y, pred_3d_z], dim=1)
        # import pdb
        # pdb.set_trace()
        norm = torch.stack([x, y, z], dim=1)
        if self.normalize:
            norm /= (self.eps + torch.norm(norm, dim=1, keepdim=True))
        return norm

    def forward(self, pred_3d, gt_3d):
        """
        shape is BCTV
        """
        errors = []
        for vec_length in self.vec_length_list:
            norm_pred = self.normal(pred_3d, vec_length)
            norm_gt = self.normal(gt_3d, vec_length)
            err = norm_pred - norm_gt
            if self.diff_order == 'L1':
                err = torch.abs(err).mean()
            elif self.diff_order == 'L2':
                err = torch.norm(err, dim=1).mean()
            errors.append(err)
        return sum(errors) / len(errors)


class MotionLoss(nn.Module):
    def __init__(self, vec_length, eps=1e-6):
        super().__init__()
        self.vec_length = vec_length
        self.eps = eps

    def motion_vec(self, data):
        B, C, T, V = data.shape
        start = data[:, :, :T - self.vec_length, :]
        end = data[:, :, self.vec_length:, :]
        x = \
            start[..., 1] * end[..., 2] - \
            start[..., 2] * end[..., 1]
        y = \
            start[..., 2] * end[..., 0] - \
            start[..., 0] * end[..., 2]
        z = \
            start[..., 0] * end[..., 1] - \
            start[..., 1] * end[..., 0]
        norm = torch.stack([x, y, z], dim=1)
        # norm = norm / (torch.sqrt(torch.sum(norm ** 2, dim=1, keepdim=True)) + self.eps)
        norm = norm / (torch.norm(norm, dim=1, keepdim=True) + self.eps)
        return norm * 1000.0

    def forward(self, pred_3d, gt_3d):
        """
        shape is BCTV
        """
        pred_3d_norm = self.motion_vec(pred_3d)
        gt_3d_norm = self.motion_vec(gt_3d)
        norm_loss = torch.abs(pred_3d_norm - gt_3d_norm)
        loss = torch.mean(norm_loss)
        return loss


def acos_safe(x, eps=1e-4):
    slope = np.arccos(1 - eps) / eps
    # TODO: stop doing this allocation once sparse gradients with NaNs (like in
    # th.where) are handled differently.
    buf = torch.empty_like(x)
    good = abs(x) <= 1 - eps
    bad = ~good
    sign = torch.sign(x[bad])
    buf[good] = torch.acos(x[good])
    buf[bad] = torch.acos(sign * (1 - eps)) - slope * sign * (abs(x[bad]) - 1 + eps)
    return buf


class CosDiff(nn.Module):
    def __init__(self, arccos=False):
        super(CosDiff, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.arccos = arccos
        self.epsilon = 1e-6
        self.pi = 3.14159265359

    def forward(self, pred, gt):
        sim = self.cos_sim(pred, gt)
        if self.arccos:
            diff = acos_safe(sim, self.epsilon).mean()
            return diff * 180.0 / self.pi
        else:
            return 1 - torch.mean(sim)


def bone_symmetric_error(predicted_3d_pos, skeleton, diff_order='L1'):
    # Shape: BCTV
    # pdb.set_trace()
    left_parent = predicted_3d_pos[..., skeleton.parents()[skeleton._joints_left]]
    left = predicted_3d_pos[..., skeleton._joints_left]
    right_parent = predicted_3d_pos[..., skeleton.parents()[skeleton._joints_right]]
    right = predicted_3d_pos[..., skeleton._joints_right]

    left_bone_lengths = torch.norm(left - left_parent, dim=1, keepdim=True)
    right_bone_lengths = torch.norm(right - right_parent, dim=1, keepdim=True)
    if diff_order == 'L1':
        return F.l1_loss(left_bone_lengths, right_bone_lengths)
    elif diff_order == 'L2':
        return torch.mean(torch.norm(left_bone_lengths - right_bone_lengths, dim=1, keepdim=True))
    elif diff_order == 'MSE':
        return F.mse_loss(left_bone_lengths, right_bone_lengths)


def mpjpe(predicted, target):
    """
    Shape: BCTV
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=1))


def mpjpe_l1(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    # return torch.mean(torch.abs(predicted - target))
    return F.l1_loss(predicted, target)


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    Shape: BCTV
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=1))


def p_mpjpe(pre, gt):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    Shape: BCTV
    """
    predicted = pre.transpose(0, 2, 3, 1)
    target = gt.transpose(0, 2, 3, 1)
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))


def n_mpjpe(pre, gt):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    Shape: BCTV
    """
    predicted = pre.transpose(0, 2, 3, 1)
    target = gt.transpose(0, 2, 3, 1)
    assert predicted.shape == target.shape

    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

# def mean_velocity_error(predicted, target):
#     """
#     Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
#     """
#     assert predicted.shape == target.shape
#
#     velocity_predicted = np.diff(predicted, axis=0)
#     velocity_target = np.diff(target, axis=0)
#
#     return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape) - 1))
