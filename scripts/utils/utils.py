#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from math import sin, cos, pi
import numpy as np
import cv2
from copy import deepcopy
from pathlib import Path
from PIL import Image

rot_psi = lambda phi: np.array([
		[1, 0, 0, 0],
		[0, np.cos(phi), -np.sin(phi), 0],
		[0, np.sin(phi), np.cos(phi), 0],
		[0, 0, 0, 1]])

rot_theta = lambda th: np.array([
		[np.cos(th), 0, -np.sin(th), 0],
		[0, 1, 0, 0],
		[np.sin(th), 0, np.cos(th), 0],
		[0, 0, 0, 1]])

rot_phi = lambda psi: np.array([
		[np.cos(psi), -np.sin(psi), 0, 0],
		[np.sin(psi), np.cos(psi), 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]])

trans_t = lambda t1, t2, t3: np.array([
		[1, 0, 0, t1],
		[0, 1, 0, t2],
		[0, 0, 1, t3],
		[0, 0, 0, 1]])

def rotmat(a, b):
	"""
	Compute the rotation matrix between two vectors.
	"""
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def add_noise_to_transform_matrix(transform_matrix, delta_phi=0, delta_theta=0, delta_psi=0, delta_tx=0, delta_ty=0,
								  delta_tz=0, delta_rot_range=0, delta_trans_range=0, RANDOM=False):
	transform_matrix_noisy = transform_matrix.copy()

	if RANDOM:
		delta_phi = np.random.uniform(-delta_rot_range, delta_rot_range)
		delta_theta = np.random.uniform(-delta_rot_range, delta_rot_range)
		delta_psi = np.random.uniform(-delta_rot_range, delta_rot_range)
		delta_tx = np.random.uniform(-delta_trans_range, delta_trans_range)
		delta_ty = np.random.uniform(-delta_trans_range, delta_trans_range)
		delta_tz = np.random.uniform(-delta_trans_range, delta_trans_range)

	# We have to decompose the transform matrix, do rotation first, then translation
	transform_matrix_temp_rot = np.eye(4)
	transform_matrix_temp_rot[:3, :3] = transform_matrix_noisy[:3, :3]
	transform_matrix_temp_trans = np.eye(4)
	transform_matrix_temp_trans[:3, 3] = transform_matrix_noisy[:3, 3]

	transform_matrix_noisy = trans_t(delta_tx, delta_ty, delta_tz) @ transform_matrix_temp_trans @ rot_phi(
		delta_phi / 180. * np.pi) @ rot_theta(delta_theta / 180. * np.pi) @ rot_psi(delta_psi / 180. * np.pi) @ transform_matrix_temp_rot

	return transform_matrix_noisy

def OpenCV2NeRF(c2w):
	"""
	Convert OpenCV camera to NeRF camera.
	OpenCV and NeRF share the same world coordinate system while NeRF uses a OpenGL camera
	"""

	c2w_new = c2w.copy()

	c2w_new[0:3, 2] *= -1  # flip the y and z axis
	c2w_new[0:3, 1] *= -1

	return c2w_new

def NeRF2OpenCV(c2w):
	"""
	Convert NeRF camera to OpenCV camera.
	OpenCV and NeRF share the same world coordinate system while NeRF uses a OpenGL camera
	"""

	c2w_new = c2w.copy()

	c2w_new[0:3, 2] *= -1  # flip the y and z axis
	c2w_new[0:3, 1] *= -1

	return c2w_new

def NeRF_vec_to_NGP_vec(vec, scale, offset):
	"""
	Convert a NeRF vector to NGP vector.
	"""
	vec = vec * scale + offset
	vec_new = vec.copy()
	vec_new[0] = vec[1]
	vec_new[1] = vec[2]
	vec_new[2] = vec[0]

	return vec_new

def NGP_vec_to_NeRF_vec(vec, scale, offset):
	"""
	Convert a NGP vector to NeRF vector.
	"""

	vec_new = vec.copy()

	vec_new[0] = vec[2]
	vec_new[2] = vec[1]
	vec_new[1] = vec[0]

	vec_new = (vec_new- offset) / scale

	return vec_new


def NeRF2NGP(c2w, scale, offset):
	"""
	Convert a NeRF matrix to NGP matrix.
	"""

	c2w_new = c2w.copy()

	c2w_new[0:3, 1] *= -1  # flip the y and z axis
	c2w_new[0:3, 2] *= -1
	c2w_new[0:3, 3] = c2w_new[0:3, 3] * scale + offset

	tmp = c2w_new[0,:].copy()
	c2w_new[0, :] = c2w_new[1, :]
	c2w_new[1, :] = c2w_new[2, :]
	c2w_new[2, :] = tmp

	return c2w_new

def NGP2NeRF(c2w, scale, offset):
	"""
	Convert a NGP matrix to NERF matrix.
	"""

	c2w_new = c2w.copy()

	tmp = c2w_new[0,:].copy()
	c2w_new[0, :] = c2w_new[2, :]
	c2w_new[2, :] = c2w_new[1, :]
	c2w_new[1, :] = tmp

	c2w_new[0:3, 1] *= -1  # flip the y and z axis
	c2w_new[0:3, 2] *= -1
	c2w_new[0:3, 3] = (c2w_new[0:3, 3] - offset) / scale

	return c2w_new

def rgb2bgr(img_rgb):
	"""
	Convert an RGB image to BGR.
	"""
	img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
	return img_bgr

def show_img(title, img_rgb):
	"""
	Show an image.
	"""
	img_bgr = rgb2bgr(img_rgb)
	cv2.imshow(title, img_bgr)
	cv2.waitKey(0)

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def process_pose(src_pose, meta_optimization):
	"""
	Convert a 4x4 pose (c2w) to 3x4 matrix (w2c). Transform it from NeRF coordinate (re-scaled/re-centered) to the original one provided by the dataset.
	"""

	pose = deepcopy(src_pose)

	pose[0:3, 3] /= meta_optimization["scale_ratio_train"]
	pose[0:3, 3] += meta_optimization["re_center_offset_train"]

	pose = NeRF2OpenCV(pose)

	pose = np.linalg.inv(pose)

	return pose[:3, :]


def create_gif(image_folder, time_acc, output_gif='output.gif'):
	"""
	Create a GIF from a list of images. The images are sorted by file name.
	"""

	# Get all files from the directory
	files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('jpeg', 'png', 'jpg'))]
	files.sort()  # To ensure they're in the order you want

	# Create the frames
	frames = [Image.open(file) for file in files]

	# Convert the time_acc from seconds to milliseconds
	total_time = time_acc * 1000

	# Calculate the duration for each frame
	duration = total_time // len(frames)  # Floor division to get whole milliseconds

	# Save frames as a GIF
	frames[0].save(output_gif, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)


