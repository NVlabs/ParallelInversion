#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import run_pose_refinement
import numpy as np
import tqdm
import argparse

scene_list = [
	"fern",
	"fortress",
	"horns",
	"room",
]

loss_type_list = [
	"Huber",
	"RelativeL2",
	"L2",
	"L1",
	"Mape",
	"Smape",
	"LogL1"
]


def shell(args):
	# No need to parallelize this since NGP will automatically make full use of the GPU computing power

	for scene in scene_list:

		print("----------------------------------------------------")
		print(f"Scene: {scene}")

		args.scene = scene
		run_pose_refinement.run_exp(args)

if __name__ == "__main__":

	parser = run_pose_refinement.config_parser()
	args = parser.parse_args()

	# Disable GUI for the modeling process
	args.gui = False

	if args.eval_mode == "main":
		# Main experiment
		args.SAVE_FOLDER = f"exp_main_nerf_llff"
		if args.noise:
			args.SAVE_FOLDER += "_noisy"
		if args.n_camera_samples > 1:
			args.SAVE_FOLDER +="_MultiGuess"
		else:
			args.SAVE_FOLDER +="_SingleGuess"

		# The default one
		args.loss_type = "L2"
		args.save_snapshot = f"model_{args.loss_type}.msgpack"

		print(f"Loss type: {args.loss_type}")

		shell(args)

	elif args.eval_mode == "ablation_rgb_loss":
		# Ablation on the loss type
		for loss_type in loss_type_list:
			print(loss_type)

			args.SAVE_FOLDER = f"exp_ablation_{loss_type}"
			if args.noise:
				args.SAVE_FOLDER += "_noisy"
			if args.n_camera_samples > 1:
				args.SAVE_FOLDER += "_MultiGuess"
			else:
				args.SAVE_FOLDER += "_SingleGuess"

			args.loss_type = loss_type
			args.save_snapshot = f"model_{loss_type}.msgpack"
			shell(args)
	else:
		raise ValueError("Unknown eval mode")
