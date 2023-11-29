#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import commentjson as json

import numpy as np

import shutil
import time

from common import *
from scenes import scenes_nerf

from tqdm import tqdm

import pyngp as ngp # noqa

import configargparse
import yaml

def config_parser():
	parser = configargparse.ArgumentParser(
		description="Run neural graphics primitives testbed with additional configuration & output options",
		config_file_parser_class=configargparse.YAMLConfigFileParser
	)

	parser.add_argument('--config', is_config_file=True, type=yaml.safe_load,
						help='config file path')

	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")
	parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', 'image' or 'volume'. Inferred from the scene if unspecified.")
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", default="", help="Load this snapshot before training. recommended extension: .msgpack")
	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .msgpack")

	parser.add_argument("--near_distance", default=-1, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
	parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")

	parser.add_argument("--sharpness_threshold", type=float, default=0, help="Set the discard threshold for sharpness in a local window.")
	parser.add_argument("--sharpen", type=float, default=0, help="Set amount of sharpening applied to NeRF training images.")

	parser.add_argument("--rendering_min_transmittance", default=1e-4, type=float, help="Control the rendering minimum transmittance.")
	parser.add_argument("--render_aabb_scale", default=1, help="Set render_aabb_scale to crop the scene")

	return parser

if __name__ == "__main__":

	parser = config_parser()
	args = parser.parse_args()

	if args.mode == "":
		if args.scene in scenes_nerf:
			args.mode = "nerf"
		else:
			raise ValueError("Must specify either a valid '--mode' or '--scene' argument.")

	if args.mode == "nerf":
		mode = ngp.TestbedMode.Nerf
		configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
		scenes = scenes_nerf
	else:
		raise ValueError("Must specify either a valid '--mode' or '--scene' argument.")

	base_network = os.path.join(configs_dir, "base.json")
	if args.scene in scenes:
		network = scenes[args.scene]["network"] if "network" in scenes[args.scene] else "base"
		base_network = os.path.join(configs_dir, network+".json")

	fpath = os.path.realpath(__file__)
	cpath = fpath.rsplit('/', 1)[0]

	network = cpath + args.network if args.network else base_network
	if not os.path.isabs(network):
		network = os.path.join(configs_dir, network)

	testbed = ngp.Testbed(mode)
	testbed.nerf.sharpness_threshold = float(args.sharpness_threshold)
	testbed.nerf.sharpen = float(args.sharpen)
	testbed.exposure = args.exposure
	testbed.nerf.rendering_min_transmittance = args.rendering_min_transmittance

	# Enable it to have a clearer view
	# testbed.display_gui = False

	# For debug
	testbed.nerf.visualize_cameras = True
	testbed.visualize_unit_cube = True
	# testbed.nerf.training.random_bg_color = False # We have to disable that for small objects & not useful all the time

	if mode == ngp.TestbedMode.Sdf:
		testbed.tonemap_curve = ngp.TonemapCurve.ACES

	if args.scene:
		if not os.path.exists(args.scene) and args.scene in scenes:
			scene_train = os.path.join(scenes[args.scene]["data_dir"], scenes[args.scene]["dataset_train"])

		testbed.load_training_data(scene_train)

	if args.gui:
		# Pick a sensible GUI resolution depending on arguments.
		sw = args.width or 1920
		sh = args.height or 1080
		while sw*sh > 1920*1080*4:
			sw = int(sw / 2)
			sh = int(sh / 2)
		testbed.init_window(sw, sh)

	if args.load_snapshot:
		print("Loading snapshot ", args.load_snapshot)
		testbed.load_snapshot(args.load_snapshot)
	else:
		testbed.reload_network_from_file(network)

	ref_transforms = {}
	if args.screenshot_transforms: # try to load the given file straight away
		print("Screenshot transforms from ", args.screenshot_transforms)
		with open(args.screenshot_transforms) as f:
			ref_transforms = json.load(f)

	testbed.shall_train = args.train if args.gui else True

	testbed.nerf.render_with_camera_distortion = True

	network_stem = os.path.splitext(os.path.basename(network))[0]

	with open(scene_train) as f:
		train_json = json.load(f)

		# Enable it for faster rendering
		# # Following https://github.com/NVlabs/instant-ngp/issues/102#issuecomment-1026752592
		# testbed.render_aabb = ngp.BoundingBox([0.0, 0.0, 0.0],
		# 									  [args.render_aabb_scale, args.render_aabb_scale,
		# 									   args.render_aabb_scale])

	if args.near_distance >= 0.0:
		print("NeRF training ray near_distance ", args.near_distance)
		testbed.nerf.training.near_distance = args.near_distance

	old_training_step = 0
	n_steps = args.n_steps

	# If we loaded a snapshot, didn't specify a number of steps, _and_ didn't open a GUI,
	# don't train by default and instead assume that the goal is to render screenshots,
	# compute PSNR, or render a video.
	if n_steps < 0 and (not args.load_snapshot or args.gui):
		n_steps = 20000

	tqdm_last_update = 0
	if n_steps > 0:
		with tqdm(desc=f"Training for {n_steps} steps", total=n_steps, unit="step") as t:
			while testbed.frame():
				if testbed.want_repl():
					repl(testbed)
				# What will happen when training is done?
				if testbed.training_step >= n_steps:
					if args.gui:
						testbed.shall_train = False
					else:
						break

				# # Render image with all the camera visualization
				# if testbed.training_step >= 1000:
				# 	p = testbed.grab_gl_pixels()
				# 	imageio.imwrite("test.png", np.flipud(p[:,:,0:3]))
				# 	exit()

				# Update progress bar
				if testbed.training_step < old_training_step or old_training_step == 0:
					old_training_step = 0
					t.reset()

				now = time.monotonic()
				if now - tqdm_last_update > 0.1:
					t.update(testbed.training_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					old_training_step = testbed.training_step
					tqdm_last_update = now

	if args.save_snapshot:
		print("Saving snapshot ", args.save_snapshot)
		testbed.save_snapshot(args.save_snapshot, False)
