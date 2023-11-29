#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import glob
import os
from pathlib import Path, PurePosixPath

import imageio
import numpy as np
import json
import sys
import math
import cv2
import os
import shutil
import subprocess
from rembg.bg import remove
from PIL import Image
import io
import tqdm

import configargparse
import yaml
from copy import deepcopy

from utils.read_llff import load_llff_data

def parse_args():
	parser = configargparse.ArgumentParser(
		description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place",
		config_file_parser_class=configargparse.YAMLConfigFileParser
	)

	parser.add_argument('--config', is_config_file=True, type=yaml.safe_load,
						help='config file path')

	# Video extraction
	parser.add_argument("--video_in", default="", help="run ffmpeg first to convert a provided video file into a set of images. uses the video_fps parameter also")
	parser.add_argument("--video_fps", type=int, default=2)
	parser.add_argument("--time_slice", default="", help="time (in seconds) in the format t1,t2 within which the images should be generated from the video. eg: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video")
	parser.add_argument("--no_run_video_extraction", action='store_true', help="do not extract images from the video (assuming they are already extracted)")

	# Salient object segmentation
	parser.add_argument("--run_segment", action='store_true', help="segment the object from the background or not")
	parser.add_argument("--video_process_mode", type=int, default=2, help="Mode 1: Process one by one; Mode 2: Process in bundle")
	parser.add_argument("--seg_images", default="segmented", help="input path to the segmented images")

	# Colmap related
	parser.add_argument("--run_colmap", action="store_true", help="run colmap first on the image folder")
	parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images")
	parser.add_argument("--colmap_db", default="colmap.db", help="colmap database filename")
	parser.add_argument("--colmap_camera_model", default="OPENCV", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL","OPENCV"], help="camera model")
	parser.add_argument("--colmap_camera_params", default="", help="intrinsic parameters, depending on the chosen model.  Format: fx,fy,cx,cy,dist")
	parser.add_argument("--images", default="images", help="input path to the images")
	parser.add_argument("--images_interval", type=int, default=1, help="pick 1 image from the given interval")
	parser.add_argument("--text", default="colmap_text", help="input path to the colmap text files (set automatically if run_colmap is used)")
	parser.add_argument("--aabb_scale", default=16, choices=["1","2","4","8","16"], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
	parser.add_argument("--skip_early", default=0, help="skip this many images from the start")
	parser.add_argument("--keep_colmap_coords", action="store_true", help="keep transforms.json in COLMAP's original frame of reference (this will avoid reorienting and repositioning the scene for preview and rendering)")

	# Support for other datasets
	parser.add_argument("--run_read_from_dataset_nerf_llff", default="store_true", help="create json directly from the nerf_llff dataset")

	# Output
	parser.add_argument("--out", default="transforms.json", help="output path")
	parser.add_argument("--vocab_path", default="", help="vocabulary tree path")
	args = parser.parse_args()
	return args

def do_system(arg):
	print(f"==== running: {arg}")
	err = os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)

def grab_frame(args):
	"""Grab an image frame from the video file."""
	frames = []

	images = args.images
	video_file = args.video_in
	fps_command = float(args.video_fps) or 1.0

	try:
		shutil.rmtree(images)
	except:
		pass
	do_system(f"mkdir {images}")

	# Get the original info
	vid = cv2.VideoCapture(video_file)
	height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
	width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = round(vid.get(cv2.CAP_PROP_FPS))

	frame_count_command = int(frame_count*fps_command/fps)

	# with slice
	time_slice_value = ""
	time_slice = args.time_slice
	if time_slice:
		start, end = time_slice.split(",")
		time_slice_value = f",select='between(t\,{start}\,{end})'"

	frame_size = int(frame_count_command * width * height * 3)

	command = [
		'ffmpeg', '-i', video_file, '-f', 'image2pipe', '-vf', f'fps={fps_command}{time_slice_value}',
		'-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-vsync', 'vfr', '-', '-loglevel', 'panic', '-hide_banner'
	]
	pipe = subprocess.Popen(
		command, stdout=subprocess.PIPE)
	current_frame = np.frombuffer(
		pipe.stdout.read(frame_size), dtype='uint8')

	if current_frame.size == frame_size:

		current_frame = current_frame.reshape(int(current_frame.size / width / height / 3), height, width, 3)
		pipe.stdout.flush()
		if frames == []:
			frames = current_frame
		else:
			frames = frames + current_frame

		warning_flag = 0
	else:
		warning_flag = 1

	return frames, warning_flag

def run_ffmpeg(args):
	if not os.path.isabs(args.images):
		# Update args.images
		args.images = os.path.join(os.path.dirname(args.video_in), args.images)
	images = args.images
	video = args.video_in
	fps = float(args.video_fps) or 1.0
	print(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")

	# # We choose yes by default
	# if (input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
	# 	sys.exit(1)
	try:
		shutil.rmtree(images)
	except:
		pass
	do_system(f"mkdir {images}")

	time_slice_value = ""
	time_slice = args.time_slice
	if time_slice:
		start, end = time_slice.split(",")
		time_slice_value = f",select='between(t\,{start}\,{end})'"
	do_system(f"ffmpeg -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}{time_slice_value}\" {images}/%04d.png")

def run_segment(args):

	out = os.path.join(os.path.dirname(args.images), args.seg_images)

	if os.path.exists(out) and os.listdir(out) and (input(f"{out} is already there. skip? (Y/n)").lower().strip() + "y")[:1] == "y":
		print(f"Skip segmentation")
	else:
		do_system(f"rembg p {args.images} {out}")

def run_colmap(args):

	db = os.path.join(os.path.dirname(args.images), args.colmap_db)

	images = args.images
	db_noext = str(Path(db).with_suffix(""))

	text = db_noext + "_text"
	sparse = db_noext + "_sparse"
	print(f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")

	# # We choose yes by default
	# if (input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
	# 	sys.exit(1)
	if os.path.exists(db):
		os.remove(db)

	if args.colmap_camera_params != "":
		do_system(f"colmap feature_extractor --ImageReader.camera_model {args.colmap_camera_model} --ImageReader.camera_params \"{args.colmap_camera_params}\" --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {db} --image_path {images}")
	else:
		do_system(f"colmap feature_extractor --ImageReader.camera_model {args.colmap_camera_model} --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {db} --image_path {images}")

	match_cmd = f"colmap {args.colmap_matcher}_matcher --SiftMatching.guided_matching=true --database_path {db}"
	if args.vocab_path:
		match_cmd += f" --VocabTreeMatching.vocab_tree_path {args.vocab_path}"
	do_system(match_cmd)
	try:
		shutil.rmtree(sparse)
	except:
		pass
	do_system(f"mkdir {sparse}")
	do_system(f"colmap mapper --database_path {db} --image_path {images} --output_path {sparse}")
	do_system(f"colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1")
	try:
		shutil.rmtree(text)
	except:
		pass
	do_system(f"mkdir {text}")
	do_system(f"colmap model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
	# Based on Laplacian
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

def qvec2rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2
		]
	])

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db):
	"""
	rays are from the center of the camera
	returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	"""
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c) ** 2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa + ta * da + ob + tb * db) * 0.5, denom

def re_pose_camera(out, offset=None, scale=None):
	"""
	Re-center and re-scale the camera to make it look at the center of the scene in a unit cube.

	Args:
		out: the original json dict.
		offset: manual offset to apply to the camera center.

	Returns:
		out: updated json dict.
	"""

	nframes = len(out["frames"])

	# find a central point they are all looking at
	print("computing center of attention...")
	totw = 0.0
	totp = np.array([0.0, 0.0, 0.0])
	for f in out["frames"]:
		mf = f["transform_matrix"][0:3, :]
		for g in out["frames"]:
			mg = g["transform_matrix"][0:3, :]
			p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
			if w > 0.01:
				totp += p * w
				totw += w
	totp /= totw
	print(totp)  # the cameras are looking at totp

	# Apply offset
	if offset is not None:
		print("Manually offset by ", offset)
		totp += offset

	# Re-center
	for f in out["frames"]:
		f["transform_matrix"][0:3, 3] -= totp

	out['re_center_offset'] = totp.astype('float').tolist()

	# Re-scale
	avglen = 0.

	for f in out["frames"]:
		avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
	avglen /= nframes
	print("avg camera distance from origin", avglen)

	if scale is not None:
		print("Manually scaling by ", scale)
		avglen*=scale

	for f in out["frames"]:
		f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized", just outside of the unit cube

	out['scale_ratio'] = 4.0 / avglen

	for f in out["frames"]:
		f["transform_matrix"] = f["transform_matrix"].tolist()

	print(nframes, "frames")

	return out

if __name__ == "__main__":
	args = parse_args()

	if not os.path.isabs(args.images):
		# Work on the video input
		args.images = os.path.join(os.path.dirname(args.video_in), args.images)
	else:
		# Work on the image input
		if args.images_interval != 1:
			folder = Path(args.images)
			src_list = [f.as_posix() for f in folder.iterdir() if any(f.match(p) for p in ("*.jpg", "*.png"))]
			new_folder = os.path.join(os.path.dirname(args.images), 'temp')
			do_system(f"mkdir {new_folder} -p")
			for i in range(0, len(src_list),args.images_interval):
				do_system(f"cp {src_list[i]} {os.path.join(new_folder,os.path.basename(src_list[i]))}")

			# Rename folders
			do_system(f"mv {args.images} {os.path.join(os.path.dirname(args.images), 'src')}")
			do_system(f"mv {new_folder} {os.path.join(os.path.dirname(args.images), 'images')}")

	# Get images from the input video
	if args.video_in != "" and args.no_run_video_extraction == False:

		if os.path.exists(args.images) and os.listdir(args.images) and (input(f"{args.images} is already there. skip? (Y/n)").lower().strip() + "y")[:1] == "y":
			# If path exists & file exists & choose to skip
			print("Skip video extraction")
		else:
			# Option 1: Process one by one
			if args.video_process_mode == 1:
				frames, _ = grab_frame(args)

				for i in tqdm.tqdm(range(frames.shape[0])):
					f = Image.fromarray(frames[i].astype('uint8'), 'RGB')
					if args.run_segment is True:
						result = remove(f)
						result.save(os.path.join(args.images, f"{str(i).zfill(4)}.png"))
					else:
						f.save(os.path.join(args.images, f"{str(i).zfill(4)}.png"))
			else:
				# Option 2: Process in bundle, need more saving space. faster.
				run_ffmpeg(args)
				if args.run_segment is True:
					run_segment(args)
	else:
		assert len(args.images) > 0, "No images provided"
		# Process the images
		if args.run_segment is True:
			run_segment(args)

	if args.run_colmap:
		run_colmap(args)

		# Interpretation of COLMAP output
		AABB_SCALE = int(args.aabb_scale)
		SKIP_EARLY = int(args.skip_early)

		IMAGE_FOLDER = args.images

		TEXT_FOLDER = os.path.join(os.path.dirname(args.images), args.text)
		OUT_PATH = os.path.join(os.path.dirname(args.images), args.out)

		print(f"outputting to {OUT_PATH}...")
		with open(os.path.join(TEXT_FOLDER, "cameras.txt"), "r") as f:
			angle_x = math.pi / 2
			for line in f:
				# 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
				# 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
				# 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443

				# OPENCV CAMERA_ID, MODEL, WIDTH, HEIGHT, fx, fy, cx, cy, k1, k2, p1, p2
				# k1, k2, p1, p2 for distortion
				# https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html

				if line[0] == "#":
					continue
				els = line.split(" ")
				w = float(els[2])
				h = float(els[3])
				fl_x = float(els[4])
				fl_y = float(els[4])
				k1 = 0
				k2 = 0
				p1 = 0
				p2 = 0
				cx = w / 2
				cy = h / 2
				if els[1] == "SIMPLE_PINHOLE":
					cx = float(els[5])
					cy = float(els[6])
				elif els[1] == "PINHOLE":
					fl_y = float(els[5])
					cx = float(els[6])
					cy = float(els[7])
				elif els[1] == "SIMPLE_RADIAL":
					cx = float(els[5])
					cy = float(els[6])
					k1 = float(els[7])
				elif els[1] == "RADIAL":
					cx = float(els[5])
					cy = float(els[6])
					k1 = float(els[7])
					k2 = float(els[8])
				elif els[1] == "OPENCV":
					fl_y = float(els[5])
					cx = float(els[6])
					cy = float(els[7])
					k1 = float(els[8])
					k2 = float(els[9])
					p1 = float(els[10])
					p2 = float(els[11])
				else:
					print("unknown camera model ", els[1])
				# fl = 0.5 * w / tan(0.5 * angle_x);
				angle_x = math.atan(w / (fl_x * 2)) * 2
				angle_y = math.atan(h / (fl_y * 2)) * 2
				fovx = angle_x * 180 / math.pi
				fovy = angle_y * 180 / math.pi

		print(f"camera:\n\tres={w, h}\n\tcenter={cx, cy}\n\tfocal={fl_x, fl_y}\n\tfov={fovx, fovy}\n\tk={k1, k2} p={p1, p2} ")

		# Output
		with open(os.path.join(TEXT_FOLDER, "images.txt"), "r") as f:
			i = 0
			bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
			out = {
				"camera_angle_x": angle_x,
				"camera_angle_y": angle_y,
				"fl_x": fl_x,
				"fl_y": fl_y,
				"k1": k1,
				"k2": k2,
				"p1": p1,
				"p2": p2,
				"cx": cx,
				"cy": cy,
				"w": w,
				"h": h,
				"aabb_scale": AABB_SCALE,
				"frames": [],
			}

			# Get sharpness & transform_matrix
			up = np.zeros(3)

			for line in f:
				line = line.strip()
				if line[0] == "#":
					continue
				i = i + 1
				if i < SKIP_EARLY * 2:
					continue
				if i % 2 == 1:
					elems = line.split(" ")  # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
					# name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
					# why is this requiring a relative path while using ^
					image_rel = os.path.relpath(IMAGE_FOLDER)
					name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
					b = sharpness(name)
					print(name, "sharpness=", b)
					image_id = int(elems[0])
					qvec = np.array(tuple(map(float, elems[1:5])))
					tvec = np.array(tuple(map(float, elems[5:8])))
					R = qvec2rotmat(-qvec)
					t = tvec.reshape([3, 1])
					m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
					c2w = np.linalg.inv(m)
					if not args.keep_colmap_coords:
						c2w[0:3, 2] *= -1  # flip the y and z axis
						c2w[0:3, 1] *= -1
						c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
						c2w[2, :] *= -1  # flip whole world upside down

						up += c2w[0:3, 1]

					frame = {"file_path": name, "sharpness": b, "transform_matrix": c2w}
					out["frames"].append(frame)

		if args.keep_colmap_coords:
			flip_mat = np.array([
				[1, 0, 0, 0],
				[0, -1, 0, 0],
				[0, 0, -1, 0],
				[0, 0, 0, 1]
			])

			for f in out["frames"]:
				f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat)  # flip cameras (it just works)

			for f in out["frames"]:
				f["transform_matrix"] = f["transform_matrix"].tolist()

			nframes = len(out["frames"])
			print(nframes, "frames")
		else:
			# don't keep colmap coords - reorient the scene to be easier to work with

			# Rotate up to be the z axis
			up = up / np.linalg.norm(up)
			print("up vector was", up)
			R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
			R = np.pad(R, [0, 1])
			R[-1, -1] = 1

			for f in out["frames"]:
				f["transform_matrix"] = np.matmul(R, f["transform_matrix"])  # rotate up to be the z axis

			# Re-pose the camera
			out = re_pose_camera(out)

		print(f"writing {OUT_PATH}")
		with open(OUT_PATH, "w") as outfile:
			json.dump(out, outfile, indent=2)

	elif args.run_read_from_dataset_nerf_llff:

		# We use some default settings here
		factor = 8
		llffhold = 8

		# We assume args.images has been updated
		OUT_PATH = os.path.join(os.path.dirname(args.images), args.out)

		# Save all the processed images to a new folder
		IMAGE_PROCESSED_FOLDER = os.path.join(os.path.dirname(args.images), "processed")
		if os.path.exists(IMAGE_PROCESSED_FOLDER):
			shutil.rmtree(IMAGE_PROCESSED_FOLDER)
		os.makedirs(IMAGE_PROCESSED_FOLDER, exist_ok=True)

		# e.g., 'fern'
		SCENE_FOLDER = os.path.split(os.path.split(args.images)[0])[1]

		# The whole dataset path
		data_dir = os.path.dirname(os.path.dirname(args.images))+'/'

		poses, hwf, images = load_llff_data(data_dir=data_dir, model_name=SCENE_FOLDER, factor=factor, recenter=True, bd_factor=.75, spherify=False,
							   INERF_ORIGINAL_POSE_GENERATION=False)


		AABB_SCALE = int(args.aabb_scale)


		h, w, focal = hwf

		h = float(h)
		w = float(w)
		focal = float(focal)

		cx = w // 2
		cy = h // 2

		k1 = 0
		k2 = 0
		p1 = 0
		p2 = 0
		fl_x = focal
		fl_y = focal

		# fl = 0.5 * w / tan(0.5 * angle_x);
		angle_x = math.atan(w / (fl_x * 2)) * 2
		angle_y = math.atan(h / (fl_y * 2)) * 2

		out = {
			"camera_angle_x": angle_x,
			"camera_angle_y": angle_y,
			"fl_x": fl_x,
			"fl_y": fl_y,
			"k1": k1,
			"k2": k2,
			"p1": p1,
			"p2": p2,
			"cx": cx,
			"cy": cy,
			"w": w,
			"h": h,
			"aabb_scale": AABB_SCALE,
			"frames": [],
		}

		for i in range(poses.shape[0]):

			target_filename = os.path.join(IMAGE_PROCESSED_FOLDER, f'{str(i).zfill(3)}.png')
			imageio.imwrite(target_filename, images[i])

			b = sharpness(target_filename)

			c2w = np.eye(4)
			c2w[:3, :] = poses[i]

			frame = {"file_path": f'./processed/{str(i).zfill(3)}.png', "sharpness": b,
					 "transform_matrix": c2w}
			out["frames"].append(frame)

		# No need to convert OpenCV cam to NeRF cam as they have done the job for us

		# # Some scenes cannot be easily re-centered in an automatic way (llff cameras are all one-face forward), so manual recenter may be needed
		# if "fortress" in args.images:
		# 	# let object far away from the camera
		# 	out = re_pose_camera(out, offset=[0,0,0.7])
		# else:
		# 	out = re_pose_camera(out)

		out = re_pose_camera(out)

		print(f"writing {OUT_PATH}")
		with open(OUT_PATH, "w") as outfile:
			json.dump(out, outfile, indent=2)

		# We have to separate the dataset into train and test set
		# We use the same split as the original dataset. every 8th frame is used for testing
		OUT_PATH_TRAIN = OUT_PATH.replace(".json", "_train.json")
		OUT_PATH_TEST = OUT_PATH.replace(".json", "_test.json")

		out_train = deepcopy(out)
		out_test = deepcopy(out)

		out_test["frames"] = out_test["frames"][::llffhold]
		out_train["frames"] = [x for x in out_train["frames"] if x not in out_test["frames"]]

		with open(OUT_PATH_TRAIN, "w") as outfile:
			json.dump(out_train, outfile, indent=2)

		with open(OUT_PATH_TEST, "w") as outfile:
			json.dump(out_test, outfile, indent=2)

	else:
		raise Exception("Unknown mode")
