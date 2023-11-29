# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Yen-Chen, Lin
# ------------------------------------------------------------------------------

# Adapted from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_llff.py

import numpy as np
import os
import imageio

########## Slightly modified version of LLFF data loading code
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
	needtoload = False
	for r in factors:
		imgdir = os.path.join(basedir, 'images_{}'.format(r))
		if not os.path.exists(imgdir):
			needtoload = True
	for r in resolutions:
		imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
		if not os.path.exists(imgdir):
			needtoload = True
	if not needtoload:
		return

	from subprocess import check_output

	imgdir = os.path.join(basedir, 'images')
	imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
	imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
	imgdir_orig = imgdir

	wd = os.getcwd()

	for r in factors + resolutions:
		if isinstance(r, int):
			name = 'images_{}'.format(r)
			resizearg = '{}%'.format(100. / r)
		else:
			name = 'images_{}x{}'.format(r[1], r[0])
			resizearg = '{}x{}'.format(r[1], r[0])
		imgdir = os.path.join(basedir, name)
		if os.path.exists(imgdir):
			continue

		print('Minifying', r, basedir)

		os.makedirs(imgdir)
		check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

		ext = imgs[0].split('.')[-1]
		args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
		print(args)
		os.chdir(imgdir)
		check_output(args, shell=True)
		os.chdir(wd)

		if ext != 'png':
			check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
			print('Removed duplicates')
		print('Done')


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
	poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
	poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
	bds = poses_arr[:, -2:].transpose([1, 0])

	img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
			if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
	sh = imageio.imread(img0).shape

	sfx = ''

	if factor is not None:
		sfx = '_{}'.format(factor)
		_minify(basedir, factors=[factor])
		factor = factor
	elif height is not None:
		factor = sh[0] / float(height)
		width = int(sh[1] / factor)
		_minify(basedir, resolutions=[[height, width]])
		sfx = '_{}x{}'.format(width, height)
	elif width is not None:
		factor = sh[1] / float(width)
		height = int(sh[0] / factor)
		_minify(basedir, resolutions=[[height, width]])
		sfx = '_{}x{}'.format(width, height)
	else:
		factor = 1

	imgdir = os.path.join(basedir, 'images' + sfx)
	if not os.path.exists(imgdir):
		print(imgdir, 'does not exist, returning')
		return

	imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
				f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
	if poses.shape[-1] != len(imgfiles):
		print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
		return

	sh = imageio.imread(imgfiles[0]).shape
	poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
	poses[2, 4, :] = poses[2, 4, :] * 1. / factor

	if not load_imgs:
		return poses, bds

	def imread(f):
		if f.endswith('png'):
			return imageio.imread(f, ignoregamma=True)
		else:
			return imageio.imread(f)

	imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
	imgs = np.stack(imgs, -1)

	print('Loaded image data', imgs.shape, poses[:, -1, 0])
	return poses, bds, imgs


def normalize(x):
	return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
	vec2 = normalize(z)
	vec1_avg = up
	vec0 = normalize(np.cross(vec1_avg, vec2))
	vec1 = normalize(np.cross(vec2, vec0))
	m = np.stack([vec0, vec1, vec2, pos], 1)
	return m


def ptstocam(pts, c2w):
	tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
	return tt


def poses_avg(poses):
	hwf = poses[0, :3, -1:]

	center = poses[:, :3, 3].mean(0)
	vec2 = normalize(poses[:, :3, 2].sum(0))
	up = poses[:, :3, 1].sum(0)
	c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

	return c2w


def recenter_poses(poses):
	poses_ = poses + 0
	bottom = np.reshape([0, 0, 0, 1.], [1, 4])
	c2w = poses_avg(poses)
	c2w = np.concatenate([c2w[:3, :4], bottom], -2)
	bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
	poses = np.concatenate([poses[:, :3, :4], bottom], -2)

	poses = np.linalg.inv(c2w) @ poses
	poses_[:, :3, :4] = poses[:, :3, :4]
	poses = poses_
	return poses

#####################

def spherify_poses(poses, bds):
	p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

	rays_d = poses[:, :3, 2:3]
	rays_o = poses[:, :3, 3:4]

	def min_line_dist(rays_o, rays_d):
		A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
		b_i = -A_i @ rays_o
		pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
		return pt_mindist

	pt_mindist = min_line_dist(rays_o, rays_d)

	center = pt_mindist
	up = (poses[:, :3, 3] - center).mean(0)

	vec0 = normalize(up)
	vec1 = normalize(np.cross([.1, .2, .3], vec0))
	vec2 = normalize(np.cross(vec0, vec1))
	pos = center
	c2w = np.stack([vec1, vec2, vec0, pos], 1)

	poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

	rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

	sc = 1. / rad
	poses_reset[:, :3, 3] *= sc
	bds *= sc
	rad *= sc

	centroid = np.mean(poses_reset[:, :3, 3], 0)
	zh = centroid[2]

	poses_reset = np.concatenate(
		[poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

	return poses_reset, bds

def load_llff_data(data_dir, model_name, factor=8, recenter=True, bd_factor=.75, spherify=False, INERF_ORIGINAL_POSE_GENERATION=False):
	poses, bds, imgs = _load_data(data_dir + str(model_name) + "/", factor=factor)  # factor=8 downsamples original imgs by 8x
	print('Loaded', data_dir + str(model_name) + "/", bds.min(), bds.max())

	# Correct rotation matrix ordering and move variable dim to axis 0
	poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
	poses = np.moveaxis(poses, -1, 0).astype(np.float32)
	images = np.moveaxis(imgs, -1, 0).astype(np.float32)
	bds = np.moveaxis(bds, -1, 0).astype(np.float32)

	# Rescale if bd_factor is provided
	sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
	poses[:, :3, 3] *= sc
	bds *= sc

	if recenter:
		poses = recenter_poses(poses)

	if spherify:
		poses, bds = spherify_poses(poses, bds)

	images = np.asarray(images * 255, dtype=np.uint8)
	poses = poses.astype(np.float32)
	hwf = poses[0,:3,-1]
	poses = poses[:,:3,:4]

	return poses, hwf, images
