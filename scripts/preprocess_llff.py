import os

scene_list = [
	"fern",
	"fortress",
	"horns",
	"room",
]

if __name__ == "__main__":

	# Run colmap2nerf.py for each scene in scene_list
	for scene in scene_list:
		print("----------------------------------------------------")
		print(f"Scene: {scene}")
		os.system(f"python colmap2nerf.py --config config/dataset/nerf_llff.yaml --images ../data/nerf/nerf_llff_data/{scene}/images_8")
