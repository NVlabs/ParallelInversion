#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import shutil
import commentjson as json
import time

from common import *
from scenes import scenes_nerf

from tqdm import tqdm

import pyngp as ngp  # noqa

import cv2
import skimage
from utils.utils import show_img, OpenCV2NeRF, to8b, \
	add_noise_to_transform_matrix, create_gif
from utils.metrics import te, re
from copy import deepcopy

from pathlib import Path

import configargparse
import yaml

def config_parser():
	parser = configargparse.ArgumentParser(
		description="Parallel Inversion of Neural Radiance Fields for Robust Pose Estimation",
		config_file_parser_class=configargparse.YAMLConfigFileParser
	)

	parser.add_argument('--config', is_config_file=True,  type=yaml.safe_load,
						help='config file path')
	parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf"],
						help="Only support nerf")
	parser.add_argument("--network", default="",
						help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", default="",
						help="Load this snapshot before training. recommended extension: .msgpack")

	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0,
						help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0,
						help="Resolution height of GUI and screenshots.")

	# Identifier
	parser.add_argument("--dataset_name", default="nerf_synthetic", choices=["nerf_synthetic", "nerf_llff"],
						help="set up the dataset name to determine how to convert the final metric.")

	# Model registration
	parser.add_argument("--scene", "--training_data", default="",
						help="The scene to load. Can be the scene's name or a full path to the training data.")
	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--train", action="store_true",
						help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--save_snapshot", default="",
						help="Save this snapshot after training. recommended extension: .msgpack")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
	parser.add_argument("--sharpness_threshold", type=float, default=0, help="Set the discard threshold for sharpness in a local window.")
	parser.add_argument("--sharpen", type=float, default=0, help="Set amount of sharpening applied to NeRF training images.")
	parser.add_argument("--render_aabb_scale",  type=float, default=1, help="set render_aabb_scale to crop the scene")
	parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")
	parser.add_argument("--rendering_min_transmittance", default=1e-4, type=float, help="Controls the rendering minimum transmittance.")

	# Target related
	parser.add_argument("--target_filename", default="", help="set the target view filename")
	parser.add_argument("--target_json", default="", help="force to read from a new json, otherwise read from the test one")

	# Noise settings
	parser.add_argument("--noise", type=str, default=None, nargs="*",
						help='options: gaussian / salt / pepper / sp / poisson')
	parser.add_argument("--sigma", type=float, default=0.01,
						help='var = sigma^2 of applied noise (variance = std)')
	parser.add_argument("--amount", type=float, default=0.05,
						help='proportion of image pixels to replace with noise (used in ‘salt’, ‘pepper’, and ‘s&p)')
	parser.add_argument("--delta_brightness", type=float, default=0.0,
						help='reduce/increase brightness of the observed image, value is in [-1...1]')
	parser.add_argument("--random_maskout", action="store_true",
						help='randomly mask out some of the image pixels to simulate occlusion')
	parser.add_argument("--n_maskout", type=int, default=5,
						help='the number of mask out holes to be created')
	parser.add_argument("--r_maskout", type=int, default=50,
						help='the radius of mask out holes to be created')

	# Optimization settings
	parser.add_argument("--loss_type", default="Huber", choices=["Huber", "RelativeL2", "L2","L1", "Mape", "Smape", "LogL1"],
						help="the RGB loss function to use for optimization")

	parser.add_argument("--n_camera_samples", type=int, default=400, help="set up the number of camera samples")
	parser.add_argument("--n_steps_pose_optimization", type=int, default=128*5,
						help='the number of steps for the pose optimization, note that it should be a product of n_steps_between_error_map_updates (default is 128)')

	parser.add_argument("--delta_rot_range_start", type=float, default=0,
						help='the range of the starting rotation angle (degree) of the camera pose guess (only meaningful when start from noisy pose)')
	parser.add_argument("--delta_trans_range_start", type=float, default=0,
						help='the range of the starting translation of the camera pose guess (only meaningful when start from noisy pose)')

	parser.add_argument("--delta_rot_range", type=float, default=0,
						help='the range of the rotation angle (degree) of the camera pose guess')
	parser.add_argument("--delta_trans_range", type=float, default=0,
						help='the range of the translation of the camera pose guess')

	# Hyperparameters
	parser.add_argument("--mask_supervision_lambda", type=float, default=0.5,
						help='the weight for mask supervision loss')
	parser.add_argument("--mask_render_ratio", type=float, default=0.5,
						help='the ratio of sampling from the mask vs. rendered region, the higher we will sample more from the mask')
	parser.add_argument("--extrinsic_learning_rate", type=float, default=0.001,
						help='the extrinsic learning rate for translation')
	parser.add_argument("--extrinsic_learning_rate_rot", type=float, default=0.001,
						help='the extrinsic learning rate for rotation')
	parser.add_argument("--extrinsic_learning_rate_base_step", type=int, default=256,
						help='the update step for extrinsic learning rate')
	parser.add_argument("--reg_weight", type=float, default=0.0001,
						help='the weight for the regularization loss')
	parser.add_argument("--n_steps_between_cam_updates", type=int, default=16,
						help='the number of steps between camera updates')
	parser.add_argument("--n_steps_between_error_map_updates", type=int, default=128,
						help='the number of steps between error map updates')
	parser.add_argument("--n_steps_first_reinitialization", type=int, default=256,
						help='the number of steps when we first reinitialize the camera guesses (it should be a product of n_steps_between_error_map_updates)')
	parser.add_argument("--n_steps_between_reinitialization", type=int, default=256,
						help='the number of steps between reinitialization (it should be a product of n_steps_between_error_map_updates)')
	parser.add_argument("--reinitialize_top_ratio", type=float, default=0.05,
						help='focus on the top ratio of the camera guesses (resample from them)')
	parser.add_argument("--delta_rot_range_resample", type=float, default=3,
						help='the range of the rotation angle (degree) of the camera pose guess when resampling')
	parser.add_argument("--delta_trans_range_resample", type=float, default=0.05,
						help='the range of the translation (degree) of the camera pose guess when resampling')

	parser.add_argument("--train_cam_with_segmented_bg", action="store_true",
						help='optimize on the background region')
	parser.add_argument("--reinitialize_cam_poses", action="store_true",
						help='reinitialize the camera poses')

	# Rendering options
	parser.add_argument("--single_view_experiment", action="store_true",
						help='mainly focus on a single view(last one), render the result & display the error during the optimization')
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots (renderings) to.")
	parser.add_argument("--n_steps_between_screenshot", type=int, default=20, help="the steps between each screenshot")
	parser.add_argument("--see_from_best_view", action="store_true",
						help='find the best view otherwise see from the last view by default')
	parser.add_argument("--n_steps_between_log", type=int, default=20,
						help="the steps between recording log")
	parser.add_argument("--rendering_bg_white", action="store_true",
						help='white background for rendering-only while by default it is black')

	# More debug options
	parser.add_argument("--cam_fixed_guess", action="store_true",
						help='simulate the camera pose guesses in a fixed way')
	# Only work when the camera pose is fixed (cam_fixed_guess option)
	parser.add_argument("--delta_phi", type=float, default=0,
						help='the range of the rotation angle (degree) of the camera pose guess')
	parser.add_argument("--delta_theta", type=float, default=0,
						help='the range of the rotation angle (degree) of the camera pose guess')
	parser.add_argument("--delta_psi", type=float, default=0,
						help='the range of the rotation angle (degree) of the camera pose guess')
	parser.add_argument("--delta_tx", type=float, default=0,
						help='the range of the translation of the camera pose guess')
	parser.add_argument("--delta_ty", type=float, default=0,
						help='the range of the translation of the camera pose guess')
	parser.add_argument("--delta_tz", type=float, default=0,
						help='the range of the translation of the camera pose guess')

	parser.add_argument("--only_phi", action="store_true",
						help='only add noise to the camera guesses in phi, note it is relative to the camera')
	parser.add_argument("--only_theta", action="store_true",
						help='only add noise to the camera guesses in theta, note it is relative to the camera')
	parser.add_argument("--only_psi", action="store_true",
						help='only add noise to the camera guesses in psi, note it is relative to the camera')
	parser.add_argument("--only_tx", action="store_true",
						help='only add noise to the camera guesses in tx, note it is relative to the camera')
	parser.add_argument("--only_ty", action="store_true",
						help='only add noise to the camera guesses in ty, note it is relative to the camera')
	parser.add_argument("--only_tz", action="store_true",
						help='only add noise to the camera guesses in tz, note it is relative to the camera')
	parser.add_argument("--VISUALIZE_UNIT_CUBE", action="store_true",
						help='visualize the unit cube')
	parser.add_argument("--POSE_ESTIMATION_ONLY", action="store_true",
						help='enable the GUI')
	parser.add_argument("--GUI_ENABLED", action="store_true",
						help='enable the GUI')
	parser.add_argument("--random_seed", type=int,default=-1,
						help='the random seed')
	parser.add_argument("--BULK_EXP", action="store_true",
						help='Run the experiment in bulk')
	parser.add_argument("--verbose", action="store_true",
						help='display debug information')

	# Experiment settings
	parser.add_argument("--exp_idx_choice", type=int, default=None,
						help='select the exact experiment index to run (mainly for head to head comparison)')
	parser.add_argument("--START_FROM_NOISY_GT",  action="store_true",
						help='start from the noisy ground truth (the first camera guess is set by arg.delta_phi and so on)')
	parser.add_argument("--SAVE_FOLDER", type=str, default="",
						help='the saving folder (help to organize the results)')
	parser.add_argument("--eval_mode", default="main", choices=["main", "ablation_rgb_loss"],
						help="different exps")

	return parser

def loss_mode_convertion(loss_mode):

	if loss_mode == 'Huber':
		return ngp.LossType.Huber
	elif loss_mode == 'RelativeL2':
		return ngp.LossType.RelativeL2
	elif loss_mode == 'L2':
		return ngp.LossType.L2
	elif loss_mode == 'L1':
		return ngp.LossType.L1
	elif loss_mode == 'Mape':
		return ngp.LossType.Mape
	elif loss_mode == 'Smape':
		return ngp.LossType.Smape
	elif loss_mode == 'LogL1':
		return ngp.LossType.LogL1
	else:
		raise ValueError('loss_mode is not available')

def init_pose_guess(meta_training, meta_optimization, args, start_pose=None):
	"""
	Initialize the camera pose guess by creating a new json file.
	"""
	with open(meta_training["scene_train"]) as f:
		# We may not need all the things from the training json file?
		ref_transforms = json.load(f)

		ref_transforms["skip_load_image"] = True

		if meta_optimization["fl_x"] is not None:
			ref_transforms["fl_x"] = meta_optimization["fl_x"]
		elif meta_optimization["camera_angle_x"] is not None:
			# ref_transforms["camera_angle_x"] = meta_optimization["camera_angle_x"]

			# Load the first image to get 'w' info
			filepath_tmp = os.path.join(meta_training["scene_dir"], ref_transforms['frames'][0]["file_path"])
			if ".png" in filepath_tmp or ".jpg" in filepath_tmp:
				w = cv2.imread(filepath_tmp)[:,:,0].shape[1]
			elif os.path.exists(filepath_tmp+'.png'):
				w = cv2.imread(filepath_tmp+'.png')[:, :, 0].shape[1]
			elif os.path.exists(filepath_tmp+'.jpg'):
				w = cv2.imread(filepath_tmp+'.jpg')[:, :, 0].shape[1]
			else:
				raise ValueError('No image found (we only support .png and .jpg for now)')

			# We have to compute "fl_x" otherwise the resolution info may be missing.
			ref_transforms["fl_x"]  = 0.5 * w / np.tan(0.5 * meta_optimization["camera_angle_x"])

		else:
			raise ValueError('fl_x or camera_angle_x is not available')

	# Pose from the target view (noise added)
	ref_transforms['frames'] = []

	if start_pose is None:
		assert meta_optimization["transform_matrix_gt"] is not None, "transform_matrix_gt is not initialized"
		start_pose = meta_optimization["transform_matrix_gt"]
		print('Start pose is initialized from the ground truth')
	else:
		assert start_pose is not None, "start_pose is not initialized"

		# We have to post-process the start_pose to make it compatible with the format of the training json
		# The basic idea is to convert the start_pose from OpenCV to NeRF, recenter and rescale

		# start_pose is c2w, PLEASE DOUBLE-CHECK
		start_pose = np.array(start_pose)
		start_pose = OpenCV2NeRF(start_pose)
		start_pose[0:3, 3] -= meta_optimization["re_center_offset_train"]
		start_pose[0:3, 3] *= meta_optimization["scale_ratio_train"]
		print("Start pose:", start_pose)

	if args.only_tx or args.only_ty or args.only_tz:
		coeff_list = np.linspace(-args.delta_trans_range, args.delta_trans_range, meta_optimization["n_camera_samples"])
	elif args.only_phi or args.only_theta or args.only_psi:
		coeff_list = np.linspace(-args.delta_rot_range, args.delta_rot_range, meta_optimization["n_camera_samples"])

	if args.START_FROM_NOISY_GT:
		print("Start pose is initialized from the noisy ground truth")
		transform_matrix_noisy = deepcopy(start_pose)

		delta_phi = args.delta_phi
		delta_theta = args.delta_theta
		delta_psi = args.delta_psi
		delta_tx = args.delta_tx
		delta_ty = args.delta_ty
		delta_tz = args.delta_tz


		# The delta_tx/y/z have the scale of the converted transform matrix (test),
		# so we need to multiply them by the scale of the original transform matrix
		# To have a fair comparison with the original dataset, we need to have this
		delta_tx *= meta_optimization["scale_ratio_test"]
		delta_ty *= meta_optimization["scale_ratio_test"]
		delta_tz *= meta_optimization["scale_ratio_test"]
		print(f"delta translation has been re-scaled by the scale_ratio_test: {meta_optimization['scale_ratio_test']}")

		transform_matrix_noisy = add_noise_to_transform_matrix(transform_matrix_noisy, delta_phi, delta_theta,
															   delta_psi, delta_tx, delta_ty, delta_tz)
		start_pose = transform_matrix_noisy

	# Save the start pose
	meta_optimization["transform_matrix_start_pose"] = start_pose

	for idx in range(0, meta_optimization["n_camera_samples"]):
		# ref_transforms['frames'][idx]['sharpness'] = 0
		transform_matrix_noisy = deepcopy(start_pose)

		if args.START_FROM_NOISY_GT and idx == 0:
			pass
		else:
			# Default noise
			delta_phi = 0
			delta_theta = 0
			delta_psi = 0
			delta_tx = 0
			delta_ty = 0
			delta_tz = 0

			if not args.cam_fixed_guess:

				if args.only_phi:
					delta_phi = coeff_list[idx]
				elif args.only_theta:
					delta_theta = coeff_list[idx]
				elif args.only_psi:
					delta_psi = coeff_list[idx]
				elif args.only_tx:
					delta_tx = coeff_list[idx] * transform_matrix_noisy[0, 0]
					delta_ty = coeff_list[idx] * transform_matrix_noisy[1, 0]
					delta_tz = coeff_list[idx] * transform_matrix_noisy[2, 0]
				elif args.only_ty:
					delta_tx = coeff_list[idx] * transform_matrix_noisy[0, 1]
					delta_ty = coeff_list[idx] * transform_matrix_noisy[1, 1]
					delta_tz = coeff_list[idx] * transform_matrix_noisy[2, 1]
				elif args.only_tz:
					delta_tx = coeff_list[idx] * transform_matrix_noisy[0, 2]
					delta_ty = coeff_list[idx] * transform_matrix_noisy[1, 2]
					delta_tz = coeff_list[idx] * transform_matrix_noisy[2, 2]
				else:
					# For multi-guess
					delta_phi = np.random.uniform(-args.delta_rot_range, args.delta_rot_range)
					delta_theta = np.random.uniform(-args.delta_rot_range, args.delta_rot_range)
					delta_psi = np.random.uniform(-args.delta_rot_range, args.delta_rot_range)
					delta_tx = np.random.uniform(-args.delta_trans_range, args.delta_trans_range)
					delta_ty = np.random.uniform(-args.delta_trans_range, args.delta_trans_range)
					delta_tz = np.random.uniform(-args.delta_trans_range, args.delta_trans_range)
			else:
				delta_phi = args.delta_phi
				delta_theta = args.delta_theta
				delta_psi = args.delta_psi
				delta_tx = args.delta_tx
				delta_ty = args.delta_ty
				delta_tz = args.delta_tz

			transform_matrix_noisy = add_noise_to_transform_matrix(transform_matrix_noisy, delta_phi, delta_theta, delta_psi, delta_tx, delta_ty, delta_tz)

		if args.verbose:
			print(f"transform_matrix_noisy: {transform_matrix_noisy}")

		frame = {}

		# we use the same image for all the frames. Since we have a skip load flag, the filename is not important anymore.
		frame["file_path"] = meta_optimization["dummy_filename"]

		# sharpness may help kill the outliers. Right now, keep all by default.
		# frame['sharpness'] = 20
		frame['transform_matrix'] = transform_matrix_noisy.tolist()

		ref_transforms['frames'].append(frame)

	with open(meta_training["scene_deploy"], "w") as outfile:
		json.dump(ref_transforms, outfile, indent=2)

def load_basic_info(meta_training, args):
	"""
	Load the basic information of the scene by recreating a .json file in python
	"""

	transform_matrix_gt = None

	with open(meta_training["scene_train"]) as f:
		ref_transforms = json.load(f)

		# Default settings
		scale_ratio_train = 1.0
		scale_ratio_test = 1.0
		re_center_offset_train = np.array([0.0, 0.0, 0.0])
		re_center_offset_test = np.array([0.0, 0.0, 0.0])
		obj_center = np.array([0.0, 0.0, 0.0])

		if 'scale_ratio' in ref_transforms:
			scale_ratio_train = ref_transforms['scale_ratio']
		if 're_center_offset' in ref_transforms:
			re_center_offset_train = np.array(ref_transforms['re_center_offset'])
		if 'obj_center' in ref_transforms:
			obj_center = np.array(ref_transforms['obj_center'])
			obj_center -= ref_transforms["re_center_offset"]
			obj_center *= ref_transforms["scale_ratio"]
		# print(f"obj_center: {obj_center}")

		if "fl_x" in ref_transforms:
			fl_x = ref_transforms['fl_x']
			camera_angle_x = None
		elif "camera_angle_x" in ref_transforms:
			camera_angle_x = ref_transforms['camera_angle_x']
			fl_x = None
		elif "fl_x" not in ref_transforms and "camera_angle_x" not in ref_transforms:
			# We take the fov of the first camera as the default.
			try:
				fl_x = ref_transforms['frames'][0]["fl_x"]
				camera_angle_x = None
			except:
				raise Exception("No camera_angle_x or fl_x in the json file")

		# Specify a file as a reference to get resolution
		dummy_filename = ref_transforms['frames'][0]['file_path']
		# By default, n_camera_samples is set up by the user
		n_camera_samples = args.n_camera_samples

		# Find the GT pose first from the test json file or target_json. Sometimes, we use the same one as in train.
		if args.target_json:
			with open(args.target_json) as f:
				test_transforms = json.load(f)
		else:
			with open(meta_training["scene_test"]) as f:
				test_transforms = json.load(f)

		if 'scale_ratio' in test_transforms:
			scale_ratio_test = test_transforms['scale_ratio']
		if 're_center_offset' in test_transforms:
			re_center_offset_test = np.array(test_transforms['re_center_offset'])

		for idx in range(len(test_transforms['frames'])):
			if Path(args.target_filename).stem == Path(test_transforms['frames'][idx]['file_path']).stem:

				print('Target view name', args.target_filename)
				print('Matched pose', test_transforms['frames'][idx]['file_path'])

				# M_c2w
				transform_matrix_gt = test_transforms['frames'][idx]['transform_matrix']
				transform_matrix_gt = np.array(transform_matrix_gt)

				# transform_matrix_gt has to be adjusted if scale/re_center_offset is different between train and test
				# It should be converted to the same scale as the train
				transform_matrix_gt[0:3, 3] /= scale_ratio_test
				transform_matrix_gt[0:3, 3] += re_center_offset_test
				transform_matrix_gt[0:3, 3] -= re_center_offset_train
				transform_matrix_gt[0:3, 3] *= scale_ratio_train

				# Get K info
				K = None

				if args.verbose:
					print('GT pose: ', transform_matrix_gt)
				break

		assert transform_matrix_gt is not None, "No GT pose found"

		meta_optimization = {
			"K": K, # for evaluation
			"transform_matrix_gt": transform_matrix_gt, # # for evaluation & currently we use it as a ref to create guesses
			"obj_center": obj_center, # for evaluation
			"n_camera_samples": n_camera_samples, # for evaluation
			"dummy_filename": dummy_filename, # as a placeholder
			"scale_ratio_train": scale_ratio_train, # for evaluation & also useful for creating guesses if they come from other resources
			"scale_ratio_test": scale_ratio_test, # for evaluation
			"re_center_offset_train": re_center_offset_train, # for evaluation & also useful for creating guesses if they come from other resources
			"re_center_offset_test": re_center_offset_test, # for evaluation
			"fl_x": fl_x, # we assume they remain the same for both train/test image
			"camera_angle_x": camera_angle_x, # a substitute for fl_x
			"log": [] # for logging single(best) view optimization results (time/rotation error/translation error)
		}

		return meta_optimization

	raise Exception(f"{meta_training['scene_train']} not found")

def load_target_view(meta_optimization, args):
	"""
	Load target view (may add some noise to simulate the real scenario, noise/brightness change/mask_out)
	"""

	# Save the original target view for debug visualization
	meta_optimization["raw_test_img"] = cv2.imread(args.target_filename)[:,:,::-1] # RGB space

	# Load the target view

	# We assume the target image has been processed to get the segmented part
	# Note that we MUST use read_image to get the right format

	obs_img = read_image(args.target_filename) # It is already in the linear format

	# For RGB image, we need to add another dimension for alpha channel
	if obs_img.shape[2] !=4:
		obs_img = np.concatenate([obs_img, np.ones([obs_img.shape[0], obs_img.shape[1], 1])], axis=2)

	if args.verbose:
		display_img = deepcopy(obs_img)
		display_img[..., 0:3] = linear_to_srgb(display_img[..., 0:3])

		show_img("Original Observed image", display_img)

	if args.noise is not None or args.delta_brightness != 0 or args.random_maskout:

		# Convert back to sRGB for further processing
		# obs_img is from 0 to 1
		obs_img[..., 0:3] = linear_to_srgb(obs_img[..., 0:3])

		# Save info for further processing
		alpha = obs_img[:, :, 3]

		# Change brightness of the observed image
		if args.delta_brightness != 0:

			obs_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2HSV)
			if args.delta_brightness < 0:
				obs_img[..., 2][obs_img[..., 2] < abs(args.delta_brightness)] = 0.
				obs_img[..., 2][obs_img[..., 2] >= abs(args.delta_brightness)] += args.delta_brightness
			else:
				lim = 1. - args.delta_brightness
				obs_img[..., 2][obs_img[..., 2] > lim] = 1.
				obs_img[..., 2][obs_img[..., 2] <= lim] += args.delta_brightness
			obs_img = cv2.cvtColor(obs_img, cv2.COLOR_HSV2RGB)
			if args.verbose:
				show_img("Observed image after brightness change", obs_img)

		obs_img_noised = deepcopy(obs_img)

		# Apply noise to the observed image
		if args.noise is not None:
			for noise in args.noise:
				if noise == 'gaussian':
					obs_img_noised = skimage.util.random_noise(obs_img_noised, mode='gaussian', var=args.sigma ** 2)
				elif noise == 's_and_p':
					obs_img_noised = skimage.util.random_noise(obs_img_noised, mode='s&p', amount=args.amount)
				elif noise == 'pepper':
					obs_img_noised = skimage.util.random_noise(obs_img_noised, mode='pepper', amount=args.amount)
				elif noise == 'salt':
					obs_img_noised = skimage.util.random_noise(obs_img_noised, mode='salt', amount=args.amount)
				elif noise == 'poisson':
					obs_img_noised = skimage.util.random_noise(obs_img_noised, mode='poisson')

				obs_img_noised = np.clip(obs_img_noised, 0, 1)

		if args.noise != '' and args.verbose:
			show_img("Observed image after adding noise", (np.array(obs_img_noised) * 255).astype(np.uint8))

		mask_combined = np.ones(alpha.shape, dtype=np.uint8) * 255
		if args.random_maskout:

			kernel = np.ones((10, 10), np.uint8)
			gradient = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, kernel)
			yy, xx = np.where(gradient == 1)

			for i in range(args.n_maskout):
				idx = np.random.choice(len(xx))
				mask = cv2.circle((alpha * 255).astype(np.uint8), (xx[idx], yy[idx]), args.r_maskout, (0, 0, 0), -1)
				mask_combined = cv2.bitwise_and(mask_combined, mask)

			if args.verbose:
				img_temp = (np.array(obs_img_noised) * 255).astype(np.uint8)
				obs_img_masked = cv2.bitwise_and(img_temp, img_temp, mask=mask_combined)

				show_img("Observed image after maskout", obs_img_masked)

			alpha = (mask_combined / 255).astype(np.float32)

		# Convert to linear color space
		if obs_img_noised.shape[2] == 4:
			obs_img_noised[..., 0:3] = srgb_to_linear(obs_img_noised[..., 0:3])

			# If we do not change brightness, we will have 4 channels, replace the last one with alpha
			obs_img_noised[..., 3:4] = alpha[..., None]

			# Premultiply alpha
			obs_img_noised[..., 0:3] *= obs_img_noised[..., 3:4]
		else:
			obs_img_noised = srgb_to_linear(obs_img_noised)

			# Currently, NGP only accepts 4-channels input
			obs_img_noised = np.concatenate((obs_img_noised, alpha[:, :, np.newaxis]), axis=2)
			# Premultiply alpha
			obs_img_noised[..., 0:3] *= obs_img_noised[..., 3:4]

		target_image = obs_img_noised
	else:
		print("No noise added to the observed image")
		target_image = obs_img

	# Save the observed image for debug visualization
	meta_optimization["obs_img"] = linear_to_srgb(target_image[..., 0:3])

	return target_image

def init_training(meta_training, args, load_snapshot=None, WITH_TRAINING=True):
	"""
	Training process for the object registration.
	"""

	testbed = ngp.Testbed(meta_training["mode"])
	testbed.nerf.sharpness_threshold = float(args.sharpness_threshold)
	testbed.nerf.sharpen = float(args.sharpen)
	testbed.exposure = float(args.exposure)
	testbed.nerf.rendering_min_transmittance = float(args.rendering_min_transmittance)
	testbed.load_training_data(meta_training["scene_train"])

	if load_snapshot:
		print("Loading snapshot ", load_snapshot)
		testbed.load_snapshot(load_snapshot)
	else:
		testbed.reload_network_from_file(meta_training["network"])

	if args.gui:
		testbed.init_window(meta_training["sw"], meta_training["sh"])

	testbed.shall_train = args.train if args.gui else True

	testbed.nerf.render_with_camera_distortion = True

	with open(meta_training["scene_train"]) as f:
		train_json = json.load(f)


	# Following https://github.com/NVlabs/instant-ngp/issues/102#issuecomment-1026752592
	# Just for visualization purpose
	testbed.render_aabb = ngp.BoundingBox([0.0, 0.0, 0.0],
										  [args.render_aabb_scale, args.render_aabb_scale,
										   args.render_aabb_scale])

	# testbed.nerf.training.random_bg_color = False # Default is True
	testbed.max_level_rand_training = True

	if WITH_TRAINING:
		old_training_step = 0
		n_steps = args.n_steps
		if n_steps < 0:
			n_steps = 100000

		if n_steps > 0:
			with tqdm(desc="Modeling", total=n_steps, unit="step") as t:
				while testbed.frame():
					if testbed.want_repl():
						repl(testbed)

					if testbed.training_step >= n_steps:
						if testbed.window_status:
							testbed.destroy_window()
						break

					# Update progress bar
					if testbed.training_step < old_training_step or old_training_step == 0:
						old_training_step = 0
						t.reset()

					t.update(testbed.training_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					old_training_step = testbed.training_step

			# Save to the source dir
			print("Saving snapshot ", meta_training["scene_model"])
			testbed.save_snapshot(meta_training["scene_model"], False)
			# time.sleep(1)
			# del testbed

	return testbed

def rendering_screenshot(testbed, cam_view_id, target_image, args, WITH_OBSERVED_BG=True):

	# Set camera
	testbed.set_camera_to_training_view(cam_view_id)

	# White background
	if args.rendering_bg_white:
		testbed.background_color = [1.0, 1.0, 1.0, 1.0]
	else:
		testbed.background_color = [0.0, 0.0, 0.0, 0.0]


	if args.dataset_name == "nerf_llff":
		image = testbed.render(target_image.shape[1], target_image.shape[0], args.screenshot_spp, True)
	else:
		image = testbed.render(target_image.shape[1] // 2, target_image.shape[0] // 2, args.screenshot_spp, True)

	# We need sRGB rather than linear one
	# Unmultiply alpha
	image[..., 0:3] = np.divide(image[..., 0:3], image[..., 3:4], out=np.zeros_like(image[..., 0:3]),
								where=image[..., 3:4] != 0)
	image[..., 0:3] = linear_to_srgb(image[..., 0:3])

	if WITH_OBSERVED_BG:

		# Change to ground truth mode
		testbed.render_groundtruth = True
		if args.dataset_name == "nerf_llff":
			target_image_rendered = testbed.render(target_image.shape[1], target_image.shape[0], args.screenshot_spp, True)
		else:
			target_image_rendered = testbed.render(target_image.shape[1] // 2, target_image.shape[0] // 2,
												   args.screenshot_spp, True)
		testbed.render_groundtruth = False

		target_image_rendered[..., 0:3] = np.divide(target_image_rendered[..., 0:3], target_image_rendered[..., 3:4],
													out=np.zeros_like(target_image_rendered[..., 0:3]),
													where=target_image_rendered[..., 3:4] != 0)
		target_image_rendered[..., 0:3] = linear_to_srgb(target_image_rendered[..., 0:3])

		# Overlay
		dst = cv2.addWeighted(to8b(image[..., 0:3]), 0.7, to8b(target_image_rendered[..., 0:3]), 0.3, 0)
	else:
		dst = to8b(image[..., 0:3])

	testbed.background_color = [0.0, 0.0, 0.0, 0.0]

	imageio.imwrite(f'{args.screenshot_dir}/{str(testbed.training_step).zfill(5)}.png', dst)

def main(args):

	# Basic settings
	if args.mode == "":
		if args.scene in scenes_nerf:
			args.mode = "nerf"
		else:
			raise ValueError("Must specify either a valid '--mode' or '--scene' argument.")

	mode = ngp.TestbedMode.Nerf
	configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
	scenes = scenes_nerf

	base_network = os.path.join(configs_dir, "base.json")
	if args.scene in scenes:
		network = scenes[args.scene]["network"] if "network" in scenes[args.scene] else "base"
		base_network = os.path.join(configs_dir, network + ".json")

	fpath = os.path.realpath(__file__)
	cpath = fpath.rsplit('/', 1)[0]

	network = cpath + args.network if args.network else base_network
	if not os.path.isabs(network):
		network = os.path.join(configs_dir, network)

	if args.scene:
		assert args.scene in scenes, f"Scene {args.scene} is not supported."

		# If we read from "dataset", we will read from all the json files. Not what we want.
		scene_dir = os.path.join(scenes[args.scene]["data_dir"], scenes[args.scene]["dataset"]) # a directory
		scene_train = os.path.join(scenes[args.scene]["data_dir"], scenes[args.scene]["dataset_train"]) # a json file for default training
		scene_test = os.path.join(scenes[args.scene]["data_dir"], scenes[args.scene]["dataset_test"]) # a json file for default test
		scene_deploy = os.path.join(scenes[args.scene]["data_dir"], 'transforms_temp.json') # a temp json file for creating candidate camera poses
		scene_model = os.path.join(scenes[args.scene]["data_dir"], args.save_snapshot)
	else:
		raise ValueError("Must specify either a valid '--scene' argument.")

	# Pick a sensible GUI resolution depending on arguments.
	sw = args.width or 1920
	sh = args.height or 1080
	while sw * sh > 1920 * 1080 * 4:
		sw = int(sw / 2)
		sh = int(sh / 2)

	if args.load_snapshot:
		if not os.path.exists(args.load_snapshot):
			raise ValueError(f"Snapshot {args.load_snapshot} does not exist.")
		else:
			scene_model = args.load_snapshot

	meta_training = {
		"mode": mode,
		"network": network,
		'sw': sw,
		'sh': sh,

		"scene_dir": scene_dir,
		"scene_train": scene_train,
		"scene_test": scene_test,
		"scene_deploy": scene_deploy,
		"scene_model": scene_model,
	}

	# Recreate the output folder for renderings
	if args.screenshot_dir:
		if os.path.exists(args.screenshot_dir):
			print('Removing existing directory:', args.screenshot_dir)
			shutil.rmtree(args.screenshot_dir)
		os.makedirs(args.screenshot_dir, exist_ok=True)

	# If not POSE_ESTIMATION_ONLY or the designated network doesn't exist, we will train the network from scratch.
	if not args.POSE_ESTIMATION_ONLY or (meta_training["scene_model"] and not os.path.exists(meta_training["scene_model"])):

		# If we have trained the model and we decide to skip it
		if os.path.exists(meta_training["scene_model"]) and (input(
			f"Basic NGP model is already there ({meta_training['scene_model']}). skip? (Y/n)").lower().strip() + "y")[:1] == "y":
			print(f"Skip training basic NGP model")

		else:
			# Force to delete the previous temporary json file
			if os.path.exists(scene_deploy):
				os.remove(scene_deploy)

			# Train to register the model & save the model
			init_training(meta_training, args, WITH_TRAINING=True)

	# Get the basic info from the training json
	meta_optimization = load_basic_info(meta_training, args)

	init_pose_guess(meta_training, meta_optimization, args)

	# Load the target view (may add some noise)
	target_image = load_target_view(meta_optimization, args)

	# Restart for pose optimization
	testbed = ngp.Testbed(mode)
	testbed.shall_train = True
	testbed.nerf.sharpen = float(args.sharpen)

	# Load the scene parameters
	# Note that we use linear color space by default
	testbed.load_training_data(meta_training["scene_deploy"])

	testbed.load_snapshot(meta_training["scene_model"])

	if args.GUI_ENABLED:
		testbed.init_window(meta_training["sw"], meta_training["sh"])
	testbed.nerf.render_with_camera_distortion = True

	if args.render_aabb_scale > 1:
		# Show the entire scene
		testbed.render_aabb = ngp.BoundingBox(
			[-args.render_aabb_scale, -args.render_aabb_scale, -args.render_aabb_scale],
			[args.render_aabb_scale, args.render_aabb_scale,
			 args.render_aabb_scale])
	else:
		testbed.render_aabb = ngp.BoundingBox([0.0, 0.0, 0.0],
											  [args.render_aabb_scale, args.render_aabb_scale,
											   args.render_aabb_scale])

	# Reset images
	if args.verbose:
		print('Replace the test images')
	with open(meta_training["scene_deploy"]) as f:
		testbed.nerf.training.set_image_all(target_image, None, 0, None, np.array([0]), 0)


	if args.verbose:
		print('Test images are replaced successfully')

	# Update training settings for pose estimation

	# Some other experimental features
	# testbed.nerf.training.train_envmap = True # It seems not helpful according to https://github.com/NVlabs/instant-ngp/discussions/290
	# testbed.nerf.training.sample_focal_plane_proportional_to_error = True # The performance is much worse than the default setting
	# testbed.nerf.training.sample_image_proportional_to_error = True

	testbed.nerf.training.random_bg_color = False # We MUST disable this option otherwise the background color may be different for different views
	testbed.max_level_rand_training = False

	# Hyperparameters
	testbed.nerf.training.loss_type = loss_mode_convertion(args.loss_type)
	testbed.nerf.training.mask_supervision_lambda = args.mask_supervision_lambda
	# Set lr to 0 at the beginning if we want to do single_view_experiment (to better render the beginning view)
	if args.single_view_experiment:
		testbed.nerf.training.extrinsic_learning_rate = 0
		testbed.nerf.training.extrinsic_learning_rate_rot = 0
	else:
		testbed.nerf.training.extrinsic_learning_rate = args.extrinsic_learning_rate
		testbed.nerf.training.extrinsic_learning_rate_rot = args.extrinsic_learning_rate_rot
	testbed.nerf.training.extrinsic_learning_rate_base_step = args.extrinsic_learning_rate_base_step
	testbed.nerf.training.extrinsic_l2_reg = args.reg_weight
	testbed.nerf.training.n_steps_between_cam_updates = args.n_steps_between_cam_updates
	testbed.nerf.training.n_steps_between_error_map_updates = args.n_steps_between_error_map_updates
	testbed.nerf.training.mask_render_ratio = args.mask_render_ratio

	# Ablation options
	testbed.nerf.training.train_cam_with_segmented_bg = args.train_cam_with_segmented_bg

	# Debug options
	testbed.visualize_unit_cube = args.VISUALIZE_UNIT_CUBE
	testbed.nerf.visualize_cameras = True
	testbed.nerf.training.verbose = args.verbose

	# The task related options
	testbed.nerf.training.optimize_extrinsics = True
	testbed.shall_train_encoding = False
	testbed.shall_train_network = False

	# Reset some parameters
	testbed.reset_loss()
	testbed.reset_training_step()

	old_training_step = 0
	DISPLAY_FLAG = True
	time_acc = 0
	end_time_others = 0
	start_time = time.time()
	with tqdm(desc="Pose optimization", total=args.n_steps_pose_optimization, unit="step") as t:

		while testbed.frame():

			start_time_others = time.time()

			if testbed.want_repl():
				repl(testbed)

			if args.verbose:

				if DISPLAY_FLAG and testbed.training_step <= args.n_steps_pose_optimization:
					print('training_step: ', testbed.training_step)

			# single_view_experiment is to focus on a specific view, render the scene, and output the pose error (mainly for demo)
			if args.single_view_experiment:

				end_time = time.time() - start_time
				time_acc += end_time_others
				compute_time = end_time - time_acc
				print('training_step: ', testbed.training_step)
				print('Computation time:', compute_time)

				if args.see_from_best_view:
					if testbed.training_step >= args.n_steps_between_error_map_updates:
						# Find the best matched pose & calculate the error
						# Note that if the error map has not been updated, there will be a problem here
						best_view_index = testbed.find_best_training_view_loss()
						print('Best matched view index:', best_view_index)
					else:
						# The error map has not been updated yet, we cannot find the best matched pose
						# Use the start pose as the best matched pose for now
						print('Warning! The error map has not been updated yet, so we use the first view as the best matched pose.')
						best_view_index = 0
				else:
					best_view_index = meta_optimization["n_camera_samples"] - 1

				# 3*4 matrix of the test pose. Has been converted to NeRF format.
				# We only checked the last camera
				test_pose = testbed.nerf.training.get_camera_extrinsics(
					best_view_index)
				test_pose = np.array(test_pose)

				# Rendering the scene
				assert args.screenshot_dir != "", "Please specify the screenshot dir"
				if (testbed.training_step) % args.n_steps_between_screenshot == 0 or testbed.training_step == 1:
					rendering_screenshot(testbed, best_view_index, target_image, args)

				if meta_optimization["transform_matrix_gt"] is not None:
					# Calculate the error
					error_rotation = re(test_pose[:3, :3], meta_optimization["transform_matrix_gt"][:3, :3])
					error_translation = te(test_pose[:3, 3], meta_optimization["transform_matrix_gt"][:3, 3])

					# Save the error for future analysis
					# time/rotation error/translation error (converted)
					if (testbed.training_step) % args.n_steps_between_log == 0 or testbed.training_step == 1:
						meta_optimization["log"].append([float(compute_time), float(error_rotation), float(error_translation/meta_optimization["scale_ratio_train"])])

					print(f'Rotation error: {error_rotation} degree')
					print(
						f'Translation error: {error_translation}/converted({error_translation / meta_optimization["scale_ratio_train"]})')

					if error_rotation < 5 and error_translation < 0.05:
						print('Success!')
					# exit()
					else:
						print('Failed!')

				# Reset the learning rate (only once) after we have rendered the scene for the first time
				if testbed.training_step == 1:
					testbed.nerf.training.extrinsic_learning_rate = args.extrinsic_learning_rate
					testbed.nerf.training.extrinsic_learning_rate_rot = args.extrinsic_learning_rate_rot

			# When training is done or we fail in the optimization process -> Find the best matched pose & calculate the error & stop the training
			if testbed.training_step >= args.n_steps_pose_optimization or \
				(testbed.training_step<args.n_steps_pose_optimization and testbed.shall_train is False and args.BULK_EXP):

				# Check if screenshot is saved => create gif
				if args.screenshot_dir and os.path.exists(args.screenshot_dir):
					# Create gif
					print('Creating gif...')
					create_gif(args.screenshot_dir, time_acc)

				# Only display once
				if DISPLAY_FLAG:

					# # Here the time includes everything
					# end_time = time.time() - start_time
					# print('Computation time:', end_time)

					# Only considering the optimization process time
					end_time = time.time() - start_time
					time_acc += end_time_others
					print('Computation time:', end_time - time_acc)

					best_view_index = None
					if testbed.training_step >= args.n_steps_between_error_map_updates:
						# Find the best matched pose & calculate the error
						# Note that if the error map has not been updated, there will be a problem here
						best_view_index = testbed.find_best_training_view_loss()
						print('Best matched view index:', best_view_index)

						# Set camera view in GUI for easier debugging
						testbed.set_camera_to_training_view(best_view_index)

						# 3*4 matrix of the best matched pose. Has been converted to NeRF format.
						best_view_pose = np.eye(4)
						best_view_pose[:3,:] = testbed.nerf.training.get_camera_extrinsics(best_view_index)
					else:
						# The error map has not been updated yet, we cannot find the best matched pose
						# Use the start pose as the best matched pose for now
						print('Warning! Optimization failed. The error map has not been updated yet, so we cannot find '
							  'the updated best matched pose.')

						best_view_index = 0
						best_view_pose = meta_optimization["transform_matrix_start_pose"]

					print('Best matched pose:', best_view_pose)

					# Save for further debugging
					meta_optimization["transform_matrix_best"] = best_view_pose

					if meta_optimization["transform_matrix_gt"] is not None:

						transform_matrix_gt = meta_optimization["transform_matrix_gt"]

						print('GT pose:', transform_matrix_gt)

						# Calculate the error
						error_rotation = re(best_view_pose[:3, :3], transform_matrix_gt[:3, :3])
						error_translation = te(best_view_pose[:3, 3], transform_matrix_gt[:3, 3])

						print(f'Rotation error: {error_rotation} degrees')

						# Some dataset does not provide the metric scale yet
						print(f'Translation error: {error_translation}/converted({error_translation/meta_optimization["scale_ratio_train"]})')

						# Currently, we only support these two exps
						if args.dataset_name == "nerf_synthetic" or args.dataset_name == "nerf_llff":

							if args.BULK_EXP:
								# Return error_rotation, error_translation (converted) and meta_optimization
								return error_rotation, error_translation/meta_optimization["scale_ratio_train"], meta_optimization

					DISPLAY_FLAG = False

				if args.GUI_ENABLED:
					testbed.shall_train = False
				else:
					break

			# Re-initialize the cam poses. Have to do it after the error map has been updated, so it needs to be +1
			if args.reinitialize_cam_poses and testbed.training_step >= args.n_steps_first_reinitialization and testbed.training_step % args.n_steps_between_reinitialization == 1:

				if args.verbose:
					print("Re-initializing the cam poses")

				n_samples = meta_optimization["n_camera_samples"]

				# Get all the losses
				pose_loss_list = []
				pose_list = []
				for idx in range(n_samples):
					pose_loss_list.append(testbed.get_training_view_loss(idx))

					pose_tmp = np.eye(4)
					pose_tmp[:3, :4] = testbed.nerf.training.get_camera_extrinsics(idx)
					pose_list.append(pose_tmp)

				# Sort the losses from the lowest to the largest
				sort_idx_list = np.argsort(pose_loss_list)

				# New groups & candidates
				# They need to cover all the samples
				assert (args.reinitialize_top_ratio * n_samples).is_integer(), "The setting of reinitialize_top_ratio needs to make sure the subsample of n_samples is integer."
				reinitialize_top_ratio = max(4/n_samples, args.reinitialize_top_ratio/(2**(testbed.training_step // args.n_steps_between_reinitialization)))
				while not (reinitialize_top_ratio * n_samples).is_integer():
					reinitialize_top_ratio *= 2

				n_groups = int(round(n_samples * reinitialize_top_ratio))
				n_candidates = int(round((1 - reinitialize_top_ratio) / reinitialize_top_ratio) + 1)

				if args.verbose:
					print(f'n_samples: {n_samples}, n_groups: {n_groups}, n_candidates: {n_candidates}')

				assert n_groups*n_candidates == n_samples, f"The numbers of groups {n_groups} and candidates {n_candidates} can not match with the number of samples {n_samples}. " \
														   f"Desired reinitialize_top_ratio is {args.reinitialize_top_ratio/(2**(testbed.training_step // args.n_steps_between_reinitialization))}"

				# Re-Initialization
				pose_list_new = [None] * n_samples  # n_samples x matrix(4x4)

				for i in range(n_groups):
					for j in range(n_candidates):

						if j == 0:
							pose_list_new[i * n_candidates + j] = np.array(pose_list)[sort_idx_list[i], :, :]
						else:
							# Add noise to the transformation matrix
							pose_list_new[i * n_candidates + j] = add_noise_to_transform_matrix(
								np.array(pose_list)[sort_idx_list[i], :, :],
								delta_rot_range=args.delta_rot_range_resample,
								delta_trans_range=args.delta_trans_range_resample, RANDOM=True)

						# Need 3x4 matrix
						testbed.nerf.training.set_camera_extrinsics(i * n_candidates + j,
																	pose_list_new[i * n_candidates + j][:-1,
																	:])

				# Since we reset the cameras, we need to force to change the learning rate externally
				weight_lr = 0.33 ** int(testbed.training_step / args.n_steps_between_cam_updates / args.extrinsic_learning_rate_base_step)
				testbed.nerf.training.extrinsic_learning_rate = args.extrinsic_learning_rate * weight_lr
				testbed.nerf.training.extrinsic_learning_rate_rot = args.extrinsic_learning_rate_rot * weight_lr

				if args.verbose:
					print('All the cam poses have been re-initialized')

			# Update progress bar
			if testbed.training_step < old_training_step or old_training_step == 0:
				old_training_step = 0
				t.reset()

			t.update(testbed.training_step - old_training_step)
			t.set_postfix(loss=testbed.loss)
			old_training_step = testbed.training_step

			end_time_others = time.time() - start_time_others

def run_exp(args):
	"""
	Experiment
	"""

	if args.random_seed != -1:
		np.random.seed(args.random_seed)

	# Multiple exps

	# Enable GUI
	# args.GUI_ENABLED = False

	if args.dataset_name == "nerf_synthetic" or args.dataset_name == "nerf_llff":

		# Use the default target_json instead
		args.target_json = ""

		with open(os.path.join(scenes_nerf[args.scene]["data_dir"], scenes_nerf[args.scene]["dataset_test"]),
				  'r') as fp:
			meta = json.load(fp)
		frames = meta['frames']

		# Save record for further analysis
		record = {
			"trial": [],

			# Noise info
			"delta_rot_range_start": float(args.delta_rot_range_start),
			"delta_trans_range_start": float(args.delta_trans_range_start),
		}

		# We will have 5 test images for each model, 5 different start poses for each
		# Initialize all the parameters at the very beginning so they remain the same for all exps
		obj_img_size = 5
		pose_init_size = 5

		obs_img_num_list = np.random.choice(len(frames), obj_img_size)
		delta_rotation_list = np.random.uniform(-args.delta_rot_range_start, args.delta_rot_range_start, size= (obj_img_size,pose_init_size,3))
		delta_translation_list = np.random.uniform(-args.delta_trans_range_start, args.delta_trans_range_start, size= (obj_img_size,pose_init_size,3))

		error_rotation_list = []
		error_translation_list = []

		trial_idx = 0

		for i, obs_img_num in enumerate(obs_img_num_list):
			for j in range(pose_init_size):
				print("*************************************")
				print(f"IDX: {trial_idx}")

				if args.exp_idx_choice is not None:
					if args.exp_idx_choice != trial_idx:
						trial_idx += 1
						continue

				# Generate random pose
				args.delta_phi = delta_rotation_list[i,j,0]
				args.delta_theta = delta_rotation_list[i,j,1]
				args.delta_psi = delta_rotation_list[i,j,2]
				args.delta_tx = delta_translation_list[i,j,0]
				args.delta_ty = delta_translation_list[i,j,1]
				args.delta_tz = delta_translation_list[i,j,2]

				print(f'delta_phi: {args.delta_phi}')
				print(f'delta_theta: {args.delta_theta}')
				print(f'delta_psi: {args.delta_psi}')
				print(f'delta_tx: {args.delta_tx}')
				print(f'delta_ty: {args.delta_ty}')
				print(f'delta_tz: {args.delta_tz}')

				if args.dataset_name == "nerf_synthetic":
					args.target_filename = os.path.join(scenes_nerf[args.scene]["data_dir"],
														frames[obs_img_num]['file_path'] + '.png')
				else:
					args.target_filename = os.path.join(scenes_nerf[args.scene]["data_dir"],
														frames[obs_img_num]['file_path'])

				error_rotation, error_translation, meta_optimization = main(args)

				error_rotation_list.append(error_rotation)
				error_translation_list.append(error_translation)

				out = {
					"trial": trial_idx,

					# Raw result
					"error_rotation": float(error_rotation),
					"error_translation": float(error_translation),

					# Start pose disturbance
					"delta_phi": args.delta_phi,
					"delta_theta": args.delta_theta,
					"delta_psi": args.delta_psi,
					"delta_tx": args.delta_tx,
					"delta_ty": args.delta_ty,
					"delta_tz": args.delta_tz,

					# Target image info
					"obs_img_num": int(obs_img_num),
					"obs_img_path": args.target_filename,

					# Raw pose info
					"gt_pose": meta_optimization["transform_matrix_gt"].tolist(),
					"start_pose": meta_optimization["transform_matrix_start_pose"].tolist(),

					# Single experiment log (will be override if multiple exps)
					"log": meta_optimization["log"],
				}

				record["trial"].append(out)

				trial_idx += 1

		record["5deg"] = float(np.mean(np.array(error_rotation_list) < 5))
		# Note that the dataset actually does not provide the metric scale yet, so 5cm only means 0.05 units in the dataset
		record["5cm"] = float(np.mean(np.array(error_translation_list) < 0.05))

		print("5deg: ", record["5deg"])
		print("5cm: ", record["5cm"])

		# Create exp folder
		timestr = time.strftime("%Y%m%d-%H%M%S")
		if args.SAVE_FOLDER:
			exp_dir = f"exp/{args.SAVE_FOLDER}/{args.dataset_name}_{str(args.scene)}_{timestr}"
		else:
			exp_dir = f"exp/{args.dataset_name}_{str(args.scene)}_{timestr}"
		os.makedirs(exp_dir, exist_ok=True)

		# Save the updated config
		with open(os.path.join(exp_dir, "config.json"), 'w') as fp:
			json.dump(vars(args), fp)

		# Save json
		with open(os.path.join(exp_dir, "record.json"), 'w') as fp:
			json.dump(record, fp)

	else:
		raise ValueError("The dataset name is not supported.")

if __name__ == "__main__":

	parser = config_parser()
	args = parser.parse_args()

	if args.random_seed != -1:
		np.random.seed(args.random_seed)

	if args.BULK_EXP:
		run_exp(args)
	else:
		# Single exp
		main(args)
