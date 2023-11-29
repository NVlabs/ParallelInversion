#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from math import sin, cos, pi
import argparse
import numpy as np

def gen_camera_pose_general(focus_center=(0, 0, 0), cam_center=(0, 0, 0), radius=1, azimuth=0, polar=pi / 3, RANDOM=False,
			   HEMISPHERE=True, NOISE=False):
	"""
    Generate camera poses on a hemisphere randomly or from a trajectory.

    Args:
        focus_center: The center of the focus.
        cam_center: The center of the generated cameras.
        radius: The distance from the camera position to the cam_center.
        azimuth: The azimuthal angle (rad).
        polar: The polar angle (rad).
        RANDOM: Whether to generate camera pose in random or not.
        HEMISPHERE: Randomly generated on a hemisphere or a sphere.
        NOISE: Whether to add noise to the radius distance.

    Returns:
        T: camera pose (M_c2w).
    """

	# Calculate camera location
	if RANDOM:
		phi = np.random.uniform(0, 2 * pi)   # azimuthal angle
		if HEMISPHERE:
			theta = np.random.uniform(0, pi / 2)  # polar angle
		else:
			theta = np.random.uniform(0, pi)  # polar angle
	else:
		phi = azimuth
		theta = polar

	if NOISE is True:
		r = radius + 0.1 * radius * np.random.uniform(-1, 1)  # 10% noise in radius
	else:
		r = radius

	x = r * sin(theta) * cos(phi)
	y = r * sin(theta) * sin(phi)
	z = r * cos(theta)

	camLoc = np.asarray([(x, y, z)]) + np.asarray([cam_center])

	# calculate camera rotation
	target_position = np.asarray([focus_center])

	# Toward the target_position
	camZ = -camLoc + target_position
	camZ = camZ / np.linalg.norm(camZ)  # normalize

	camX = np.cross(np.array([0, 0, -1]), camZ)
	camX = camX / np.linalg.norm(camX)  # normalize

	camY = np.cross(camZ, camX)
	camY = camY / np.linalg.norm(camY)  # normalize

	# Construct camera pose matrix
	T = np.eye(4)

	T[:3, 0] = camX
	T[:3, 1] = camY
	T[:3, 2] = camZ
	T[:3, 3] = camLoc

	return T

def gen_camera_pose_hemisphere(n_phi, n_theta, focus_center=(0.,0.,0.), cam_center=(0.,0.,0.), radius=1, outfile_name=''):
	"""Serve as a wrapper"""

	# Setup
	phi = np.linspace(0,2*pi, num=n_phi, endpoint=False)
	# Less dense on the pole area https://stackoverflow.com/a/33977070
	theta = np.arccos(np.linspace(1e-2,1-1e-2, num=n_theta))

	grid_phi, grid_theta = np.meshgrid(phi,theta, indexing='ij')

	poses = []
	for j in range(n_theta):
		for i in range(n_phi):
			poses.append(gen_camera_pose_general(focus_center=focus_center, cam_center=cam_center, radius=radius, azimuth=grid_phi[i,j], polar=grid_theta[i,j]))

	# Save
	if outfile_name!='':
		np.save(outfile_name, np.array(poses))

	return np.array(poses)

def gen_camera_pose_sphere(n_phi,n_theta, focus_center=(0.,0.,0.), cam_center=(0.,0.,0.), radius=1, outfile_name=''):
	"""Serve as a wrapper"""

	# Setup
	phi = np.linspace(0,2*pi, num=n_phi, endpoint=False)
	# Less dense on the pole area https://stackoverflow.com/a/33977070
	theta = np.arccos(np.linspace(-1+1e-2,1-1e-2, num=n_theta))

	grid_phi, grid_theta = np.meshgrid(phi,theta, indexing='ij')

	poses = []
	for j in range(n_theta):
		for i in range(n_phi):
			poses.append(gen_camera_pose_general(focus_center=focus_center, cam_center=cam_center, radius=radius, azimuth=grid_phi[i,j], polar=grid_theta[i,j]))

	# Save
	if outfile_name!='':
		np.save(outfile_name, np.array(poses))

	return np.array(poses)

if __name__  == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--n_phi', default=25, help="Number of azimuth interval")
	parser.add_argument('--n_theta', default=25, help="Number of polar interval")
	parser.add_argument('--focus_center', default=(0, 0, 0), help="Where the camera looks at")
	parser.add_argument('--cam_center', default=(0, 0, 0), help="Where the camera is generated around")
	parser.add_argument('--radius', default=1, help="The distance from the camera position to the cam_center")
	parser.add_argument('--outfile_name',type=str, default='sphere.npy', help="Path to where the final files will be saved")
	parser.add_argument('--verbose', action='store_true', help="Whether to debug")
	args = parser.parse_args()

	# Setup
	phi = np.linspace(0,2*pi, num=args.n_phi)
	# Less dense on the pole area https://stackoverflow.com/a/33977070
	theta = np.arccos(np.linspace(-1+1e-2,1-1e-2, num=args.n_theta))

	grid_phi, grid_theta = np.meshgrid(phi,theta, indexing='ij')

	poses = []
	for j in range(args.n_theta):
		for i in range(args.n_phi):
			poses.append(gen_camera_pose_general(focus_center=args.focus_center, cam_center=args.cam_center, radius=args.radius, azimuth=grid_phi[i,j], polar=grid_theta[i,j]))

	# Save
	np.save(args.outfile_name, np.array(poses))

	if args.verbose:

		import matplotlib.pyplot as plt

		fig = plt.figure(figsize=(7, 7))
		ax = fig.gca(projection='3d')
		ax.set_aspect("auto")
		ax.set_xlim([-2, 2])
		ax.set_ylim([-2, 2])
		ax.set_zlim([-2, 2])
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')

		for idx, T in enumerate(poses):
			if 1:
				# ax.scatter(T[0,3], T[1,3], T[2,3])
				ax.quiver(T[0,3], T[1,3], T[2,3], T[0,0], T[1,0], T[2,0], length=0.1, normalize=True, color='r')
				ax.quiver(T[0,3], T[1,3], T[2,3], T[0,1], T[1,1], T[2,1], length=0.1, normalize=True, color='g')
				ax.quiver(T[0,3], T[1,3], T[2,3], T[0,2], T[1,2], T[2,2], length=0.1, normalize=True, color='b')

		plt.show()
