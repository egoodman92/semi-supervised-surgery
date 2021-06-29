from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import json

from PIL import Image
from pycocotools.coco import COCO

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2

import sys
sys.path.append("../deep-high-resolution-net.pytorch/lib")
import time

# import _init_paths
import models
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform
from utils.score_utils import match_bboxes

import json
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from scipy import stats




_BLACK = (0, 0, 0)
_RED = (0, 0, 255)
_BLUE = (255, 0, 0) 
_PURPLE = (204, 0, 153)
_ORANGE = (51, 153, 255)
_LBROWN = (0, 153, 230)
keypoint_colors = { '1': _RED, '2': _RED, '3': _RED, '4': _RED, '5': _RED,
							'6': _ORANGE, '7': _ORANGE, '8': _ORANGE, '9': _ORANGE, 
							'10': _LBROWN, '11': _LBROWN, '12': _LBROWN, '13': _LBROWN,
							'14': _BLUE, '15': _BLUE, '16': _BLUE, '17': _BLUE,
							'18': _PURPLE, '19': _PURPLE, '20': _PURPLE, '21': _PURPLE
							}


gt = json.load(open("constants/coco_validated_full_8_20.json", "r"))


# size distribution
num_anns = 0
xs = 0
s = 0
m = 0
l = 0

areas = sorted([ann["area"] for ann in gt["annotations"]])
first = np.median(areas[:len(areas)//2])
second = np.median(areas)
third = np.median(areas[len(areas)//2:])

print(first, second, third)


xs_area = 0
s_area = 0
m_area = 0
l_area = 0
for ann in gt["annotations"]:
	num_anns += 1
	if ann["area"] < first:
		xs += 1
		xs_area += ann["area"]
	elif ann["area"] < second:
		s += 1
		s_area += ann["area"]
	elif ann["area"] < third:
		m += 1
		m_area += ann["area"]
	else:
		l += 1
		l_area += ann["area"]

# print("xs ", xs)
# print("small ", s)
# print("medium ",  m)
# print("large ", l)
# print("total ", num_anns)


# avg distances
backbone_sum = np.zeros((20, 5))
backbone_distances = [[[] for i in range(5)] for j in range(20)]
counts = np.zeros((20, 5))
skeleton = [[1, 2], [2, 3], [3, 4], [4, 5],[6, 7],
			[7, 8], [8, 9], [10, 11], [11, 12],
			[12, 13], [14, 15], [15, 16], [16, 17],
			[18, 19], [19, 20], [20, 21], [1, 6],
			[1, 10], [1, 14], [1, 18]]

for ann in gt["annotations"]:
	idx = None
	if ann["area"] < first:
		idx = 0
	elif ann["area"] < second:
		idx = 1
	elif ann["area"] < third:
		idx = 2
	else:
		idx = 3
 
	for i, (j1, j2) in enumerate(skeleton):
		x1, y1, vis1 = ann["keypoints"][(j1 - 1) * 3], ann["keypoints"][(j1 - 1) * 3 + 1], ann["keypoints"][(j1 - 1) * 3 + 2]
		x2, y2, vis2 = ann["keypoints"][(j2 - 1) * 3], ann["keypoints"][(j2 - 1) * 3 + 1], ann["keypoints"][(j2 - 1) * 3 + 2]

		# both keypoints are visible
		if vis1 > 0 and vis2 > 0:
			dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
			counts[i, 4] += 1
			backbone_sum[i, 4] += dist
			counts[i, idx] += 1
			backbone_sum[i, idx] += dist
			backbone_distances[i][4].append(dist)
			backbone_distances[i][idx].append(dist)


# remove outliers from average calculation
for i in range(20):
	for j in range(5):
		dists = sorted(backbone_distances[i][j])
		half = len(dists) // 2
		Q1 = np.median(dists[:half])
		if len(dists) % 2 == 0:
			Q3 = np.median(dists[half:])
		else:
			Q3 = np.median(dists[half + 1:])
		IQR = stats.iqr(dists, interpolation='midpoint')

		removed = set(dist for dist in dists if dist <= (Q1 - (1.5 * IQR)) or dist >= (Q3 + (1.5 * IQR)))

		if len(removed) > 0:
			counts[i, j] -= len(removed)
			backbone_sum[i, j] -= sum(removed)
			backbone_distances[i][j] = [d for d in dists if d not in removed]


# avg calculation 
avg_distances = backbone_sum / counts


min_thresholds = np.zeros((20, 5))
max_thresholds = np.zeros((20, 5))
# discard outliers when calculating threshold
# threshold is s.t. gt will always be marked as accurate (except for any outliers)
for i in range(20):
	for j in range(5):
		max_thresholds[i, j] = max(backbone_distances[i][j]) - avg_distances[i, j]
		min_thresholds[i, j] = min(backbone_distances[i][j]) - avg_distances[i, j]


# SAVE THRESHOLDS
pkl.dump(max_thresholds, open("constants/max_thresholds.pkl", "wb"))
pkl.dump(min_thresholds, open("constants/min_thresholds.pkl", "wb"))
pkl.dump(avg_distances, open("constants/avg_distances.pkl", "wb"))

# Other thresholds that were experimented with

		# max_thresholds[i, j] = (max(backbone_distances[i][j]) - avg_distances[i, j] + min(backbone_distances[i][j]) - avg_distances[i, j]) / 2

# thresholds = np.zeros((20, 5))
# for i in range(20):
# 	for j in range(5):
# 		thresholds[i, j] = np.std(np.asarray(backbone_distances[i][j])) * 2


# # test metric out: determine gt accuracy 
# accurate = np.zeros((20, 5))
# total = np.zeros((20, 5))
# for ann in gt["annotations"]:
# 	idx = None
# 	if ann["area"] < first:
# 		idx = 0
# 	elif ann["area"] < second:
# 		idx = 1
# 	elif ann["area"] < third:
# 		idx = 2
# 	else:
# 		idx = 3

# 	for i, (j1, j2) in enumerate(skeleton):
# 		x1, y1, vis1 = ann["keypoints"][(j1 - 1) * 3], ann["keypoints"][(j1 - 1) * 3 + 1], ann["keypoints"][(j1 - 1) * 3 + 2]
# 		x2, y2, vis2 = ann["keypoints"][(j2 - 1) * 3], ann["keypoints"][(j2 - 1) * 3 + 1], ann["keypoints"][(j2 - 1) * 3 + 2]

# 		# both keypoints are visible
# 		if vis1 > 0 and vis2 > 0:
# 			dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
# 			total[i, 4] += 1
# 			total[i, idx] += 1
# 			# if dist - avg_distances[i, idx] < max_thresholds[i, idx] and dist - avg_distances[i, idx] > min_thresholds[i, idx]:
# 			# 	accurate[i, idx] += 1
# 			# if dist - avg_distances[i, 4] < max_thresholds[i, 4] and dist - avg_distances[i, 4] > min_thresholds[i, 4]:
# 			# 	accurate[i, 4] += 1
# 			if dist < avg_distances[i, idx] + thresholds[i, idx] and dist > avg_distances[i, idx] - thresholds[i, idx]:
# 				accurate[i, idx] += 1
# 			if dist < avg_distances[i, 4] + thresholds[i, 4] and dist > avg_distances[i, 4] - thresholds[i, 4]:
# 				accurate[i, 4] += 1


# gt_accuracy = accurate / total
# print(gt_accuracy)




# EXPLORE THRESHOLDING ON VALIDATION SET

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# find % accuracy in validation predictions

def get_pose_estimation_prediction(pose_model, image, centers, scales, transform):
	rotation = 0

	# pose estimation transformation
	model_inputs = []
	for center, scale in zip(centers, scales):
		trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
		# Crop smaller image of people
		model_input = cv2.warpAffine(
			image,
			trans,
			(int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
			flags=cv2.INTER_LINEAR)

		# hwc -> 1chw
		model_input = transform(model_input)#.unsqueeze(0)
		model_inputs.append(model_input)

	# n * 1chw -> nchw
	model_inputs = torch.stack(model_inputs)

	# compute output heatmap
	output = pose_model(model_inputs.to(CTX))
	coords, _ = get_final_preds(
		cfg,
		output.cpu().detach().numpy(),
		np.asarray(centers),
		np.asarray(scales))

	return coords


def box_to_center_scale(box, model_image_width, model_image_height):
	"""convert a box to center,scale information required for pose transformation
	Parameters
	----------
	box : list of tuple
		list of length 2 with two tuples of floats representing
		bottom left and top right corner of a box
	model_image_width : int
	model_image_height : int
	Returns
	-------
	(numpy array, numpy array)
		Two numpy arrays, coordinates for the center of the box and the scale of the box
	"""
	center = np.zeros((2), dtype=np.float32)

	x1, y1, box_width, box_height = box
	center[0] = x1 + box_width * 0.5
	center[1] = y1 + box_height * 0.5

	aspect_ratio = model_image_width * 1.0 / model_image_height
	pixel_std = 200

	if box_width > aspect_ratio * box_height:
		box_height = box_width * 1.0 / aspect_ratio
	elif box_width < aspect_ratio * box_height:
		box_width = box_height * aspect_ratio
	scale = np.array(
		[box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
		dtype=np.float32)
	if center[0] != -1:
		scale = scale * 1.25

	return center, scale


def parse_args():
	parser = argparse.ArgumentParser(description='Compute Hand Skeleton Accuracy Thresholds')
	# general
	parser.add_argument('--cfg', type=str, required=True)
	parser.add_argument('--alpha', type=float, default=0.5)
	parser.add_argument('--strict', action='store_true')
	parser.add_argument('--test', action='store_true')
	parser.add_argument('opts',
						help='Modify config options using the command-line',
						default=None,
						nargs=argparse.REMAINDER)

	args = parser.parse_args()

	# args expected by supporting codebase
	args.modelDir = ''
	args.logDir = ''
	args.dataDir = ''
	args.prevModelDir = ''
	return args


def main():

	accurate = np.zeros((20, 5))
	total = np.zeros((20, 5))
	keypoint_accuracy = np.zeros((21,))
	keypoint_count = np.zeros((21,))

	os.makedirs('val_evaluation', exist_ok=True)

	# transformation
	pose_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
							 std=[0.229, 0.224, 0.225]),
	])

	# cudnn related setting
	cudnn.benchmark = cfg.CUDNN.BENCHMARK
	torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
	torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

	args = parse_args()
	update_config(cfg, args)

	pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
		cfg, is_train=False
	)

	if cfg.TEST.MODEL_FILE:
		print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
		pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
	else:
		print('expected model defined in config at TEST.MODEL_FILE')

	pose_model.to(CTX)
	pose_model.eval()
	
	ending = 'val'
	if args.test:
		ending = 'test'

	gt = COCO("../deep-high-resolution-net.pytorch/data/coco/annotations/hand_keypoints_" + ending + ".json")
	num_removed = 0
	total_keypoints = 0
	pck_distances = []
	pck_bbox_norms = []
	for img_id in gt.getImgIds():

		path = "../deep-high-resolution-net.pytorch/data/coco/images/all_images/" + gt.loadImgs([img_id])[0]['file_name']
 
		img = cv2.imread(path)
		image_pose = img.copy()

		centers = []
		scales = []

		anns = gt.loadAnns(gt.getAnnIds(imgIds=[img_id]))

		for box in [a["bbox"] for a in anns]:
			center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
			centers.append(center)
			scales.append(scale)

		# no bbox on img?
		if len(scales) == 0:
			continue

		pose_preds = get_pose_estimation_prediction(pose_model, image_pose, centers, scales, transform=pose_transform)

		# # N x 21 x 2
		keypoints = np.asarray([a["keypoints"] for a in anns]).reshape((-1, 21, 3))
		visibility = keypoints[:, :, 2]
		keypoints = keypoints[:, :, :2]


		for ann_idx, ann in enumerate(anns):
			idx = None
			if ann["area"] < first:
				idx = 0
			elif ann["area"] < second:
				idx = 1
			elif ann["area"] < third:
				idx = 2
			else:
				idx = 3

			acc, not_acc = set(), set()
			keypoint_present = set()
			for i, (j1, j2) in enumerate(skeleton):
				x1, y1 = pose_preds[ann_idx, (j1 - 1), :]
				x2, y2 = pose_preds[ann_idx, (j2 - 1), :]

				if x1 > 0 and y1 > 0:
					keypoint_present.add(j1)
				if x2 > 0 and y2 > 0:
					keypoint_present.add(j2)

				# both keypoints are visible
				if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
					dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
					total[i, 4] += 1
					total[i, idx] += 1

					if dist - avg_distances[i, idx] < max_thresholds[i, idx] and dist - avg_distances[i, idx] > min_thresholds[i, idx]:
						accurate[i, idx] += 1

						acc.add(j1)
						acc.add(j2)

					else:
						not_acc.add(j1)
						not_acc.add(j2)
					
					if dist - avg_distances[i, 4] < max_thresholds[i, 4] and dist - avg_distances[i, 4] > min_thresholds[i, 4]:
						accurate[i, 4] += 1


			# STRICT = ONLY PLACE IF KEYPOINT HAS SUCCEEDED IN ALL PARTS OF SKELETON (both sides are marked as accurate)
			# NOT STRICT = ONLY DELETE KEYPOINT IF BOTH SIDES ARE MARKED AS ACCURATE

			for ii in range(21):
				color = _BLACK
				if args.strict and (ii + 1 not in not_acc):
					keypoint_accuracy[ii] += 1
					color = keypoint_colors[str(ii + 1)]
				elif not args.strict and (ii + 1 in acc):
					keypoint_accuracy[ii] += 1
					color = keypoint_colors[str(ii + 1)]
				else:
					pose_preds[ann_idx, ii, :] = 0
					keypoints[ann_idx, ii, :] = 0
					num_removed += 1
				cv2.circle(img, (int(pose_preds[ann_idx, ii, 0]), int(pose_preds[ann_idx, ii, 1])), 4, color, -1)
				cv2.putText(img, str(ii + 1), (int(pose_preds[ann_idx, ii, 0] - 4), int(pose_preds[ann_idx, ii, 1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
				
				if ii + 1 in keypoint_present:
					keypoint_count[ii] += 1


		cv2.imwrite("val_evaluation/" + gt.loadImgs([img_id])[0]['file_name'], img)

		distances = np.sqrt(np.sum((keypoints - pose_preds) ** 2, axis=2))
		bbox = [max(a["bbox"][2], a["bbox"][3]) for a in anns]
		pck_distances.append(distances) 
		pck_bbox_norms.extend(bbox)

		visibility[visibility >= 1] = 1
		total_keypoints += np.sum(visibility)


	val_accuracy = accurate / total
	print(val_accuracy)

	keypoint_acc = keypoint_accuracy / keypoint_count
	print("Keypoint Accuracy per keypoint", keypoint_acc)
	print("Avg Keypoint Accuracy", np.sum(keypoint_accuracy) / np.sum(keypoint_count))
	print("Avg Keypoint Accuracy Thumb", np.sum(keypoint_accuracy[1:5]) / np.sum(keypoint_count[1:5]))
	print("Avg Keypoint Accuracy Pointer", np.sum(keypoint_accuracy[5:9]) / np.sum(keypoint_count[5:9]))
	print("Avg Keypoint Accuracy Middle", np.sum(keypoint_accuracy[9:13]) / np.sum(keypoint_count[9:13]))
	print("Avg Keypoint Accuracy Ring", np.sum(keypoint_accuracy[13:17]) / np.sum(keypoint_count[13:17]))
	print("Avg Keypoint Accuracy Pinky", np.sum(keypoint_accuracy[17:21]) / np.sum(keypoint_count[17:21]))
	print("Num Keypoints Removed ", num_removed, "Total Keypoints Predicted ", np.sum(keypoint_count), "Total GT Keypoints Labeled ", total_keypoints)

	# Show increase in PCK of remaining keypoints
	distances = np.vstack(pck_distances)
	gt_reference = np.stack([np.asarray(pck_bbox_norms)] * 21, axis=-1)

	pck = np.nanmean(distances <= args.alpha * gt_reference, axis=0)

	print("PCK@" + str(args.alpha)+ " per keypoint", pck)
	print("Avg PCK@" + str(args.alpha), np.nanmean(distances <= args.alpha * gt_reference))
	print("PCK@" + str(args.alpha) + " Thumb", np.nanmean(distances[:, 1:5] <= args.alpha * gt_reference[:, 1:5]))
	print("PCK@" + str(args.alpha) + " Pointer", np.nanmean(distances[:, 5:9] <= args.alpha * gt_reference[:, 5:9]))
	print("PCK@" + str(args.alpha)+ " Middle", np.nanmean(distances[:, 9:13] <= args.alpha * gt_reference[:, 9:13]))
	print("PCK@" + str(args.alpha) + " Ring", np.nanmean(distances[:, 13:17] <= args.alpha * gt_reference[:, 13:17]))
	print("PCK@" + str(args.alpha) + " Pinky", np.nanmean(distances[:, 17:21] <= args.alpha * gt_reference[:, 17:21]))
	print("PCK@" + str(args.alpha) + " Wrist", np.nanmean(distances[:, 0] <= args.alpha * gt_reference[:, 0]))



if __name__ == '__main__':
	main()
