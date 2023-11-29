#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Adapted from https://github.com/NVlabs/DeepIM-PyTorch/blob/144d92fa13b76d849dae7a4c1e8fb79da9e46970/lib/utils/pose_error.py#L109-L134"""

import numpy as np
import math

def re(R_est, R_gt):
    """
    Rotational Error.
    :param R_est: Rotational element of the estimated pose (3x3 matrix).
    :param R_gt: Rotational element of the ground truth pose (3x3 matrix).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(R_est.shape == R_gt.shape == (3, 3))
    error_cos = 0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0)
    error_cos = min(1.0, max(-1.0, error_cos)) # Avoid invalid values due to numerical errors
    error = math.acos(error_cos)
    error = 180.0 * error / np.pi # [rad] -> [deg]
    return error

def te(t_est, t_gt):
    """
    Translational Error.
    :param t_est: Translation element of the estimated pose (3x1 vector).
    :param t_gt: Translation element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(t_est.size == t_gt.size == 3)
    error = np.linalg.norm(t_gt - t_est)
    return error
