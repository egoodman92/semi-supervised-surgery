from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import re
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
import numpy as np

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

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import random

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import pickle as pkl

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

COCO_INSTANCE_CATEGORY_NAMES = [
	'__background__', 'hand'
]

# BBOX size quartiles
FIRST = 29887.783789515226 
SECOND = 68465.90732899995 
THIRD = 146222.138836773

skeleton = [[1, 2], [2, 3], [3, 4], [4, 5],[6, 7],
			[7, 8], [8, 9], [10, 11], [11, 12],
			[12, 13], [14, 15], [15, 16], [16, 17],
			[18, 19], [19, 20], [20, 21], [1, 6],
			[1, 10], [1, 14], [1, 18]]


def get_person_detection_boxes(model, img, threshold=0.5):
	pil_image = Image.fromarray(img)  # Load the image
	transform = transforms.Compose([transforms.ToTensor()])  # Defing PyTorch Transform
	transformed_img = transform(pil_image)  # Apply the transform to the image
	pred = model([transformed_img.to(CTX)])  # Pass the image to the model

	# Use the first detected person
	pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
					for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
	pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
				  for i in list(pred[0]['boxes'].cpu().detach().numpy())]  # Bounding boxes
	pred_scores = list(pred[0]['scores'].cpu().detach().numpy())

	person_boxes = []
	# Select box has score larger than threshold and is person
	for pred_class, pred_box, pred_score in zip(pred_classes, pred_boxes, pred_scores):
		if (pred_score > threshold) and pred_class == 'hand':
			person_boxes.append(pred_box)

	return person_boxes


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


def box_to_center_scale(box, model_image_width, model_image_height, gt):
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

	if gt:
		x1, y1, box_width, box_height = box
	else:
		x1, y1, x2, y2 = box
		box_width = x2 - x1
		box_height = y2 - y1

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
	parser = argparse.ArgumentParser(description='Keypoint Skeleton Accuracy by Category')
	# general
	parser.add_argument('--cfg', type=str, required=True)
	parser.add_argument('--alpha', type=float, default=0.5)
	parser.add_argument('--strict', action='store_true')
	parser.add_argument('--use_gt', action='store_true')
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

	if not args.use_gt:
		bbox_cfg = get_cfg()
		bbox_cfg.merge_from_file("../bbox.yaml")
		bbox_cfg.MODEL.WEIGHTS = "../mask-rcnn/models/bbox_final.pth"
		box_model = DefaultPredictor(bbox_cfg)

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

	attrs = json.load(open("constants/attributes.json", "r"))

	max_thresholds = pkl.load(open("constants/max_thresholds.pkl", "rb"))
	min_thresholds = pkl.load(open("constants/min_thresholds.pkl", "rb"))
	avg_distances = pkl.load(open("constants/avg_distances.pkl", "rb"))


	# zoom_factor is labeled at the image level, others are labled at the video level
	pck_distances = {'quality': {'good': [], 'okay': [], 'bad': []},
			'moving_camera': {'moving': [], 'static': []},
			'style': {'overhead': [], 'non_overhead': []},
			'severe_jumps': {'yes':[], 'no':[]},
			'zoom_factor': {'hand_level': [], 'closeup': [], 'body_level': []}}
	pck_bbox_norms = {'quality': {'good': [], 'okay': [], 'bad': []},
			'moving_camera': {'moving': [], 'static': []},
			'style': {'overhead': [], 'non_overhead': []},
			'severe_jumps': {'yes':[], 'no':[]},
			'zoom_factor': {'hand_level': [], 'closeup': [], 'body_level': []}}

	keypoint_accuracy_cats = {'quality': {'good': np.zeros((21,)), 'okay': np.zeros((21,)), 'bad': np.zeros((21,))},
			'moving_camera': {'moving': np.zeros((21,)), 'static': np.zeros((21,))},
			'style': {'overhead': np.zeros((21,)), 'non_overhead': np.zeros((21,))},
			'severe_jumps': {'yes':np.zeros((21,)), 'no':np.zeros((21,))},
			'zoom_factor': {'hand_level': np.zeros((21,)), 'closeup': np.zeros((21,)), 'body_level': np.zeros((21,))}}
	keypoint_count_cats = {'quality': {'good': np.zeros((21,)), 'okay': np.zeros((21,)), 'bad': np.zeros((21,))},
			'moving_camera': {'moving': np.zeros((21,)), 'static': np.zeros((21,))},
			'style': {'overhead': np.zeros((21,)), 'non_overhead': np.zeros((21,))},
			'severe_jumps': {'yes':np.zeros((21,)), 'no':np.zeros((21,))},
			'zoom_factor': {'hand_level': np.zeros((21,)), 'closeup': np.zeros((21,)), 'body_level': np.zeros((21,))}}

	num_removed = 0
	total_keypoints = 0
	for img_id in gt.getImgIds():
		img_name = gt.loadImgs([img_id])[0]['file_name']
		path = "../deep-high-resolution-net.pytorch/data/coco/images/all_images/" + img_name
 
		img = cv2.imread(path)
		image_debug = img.copy()
		image_pose = img.copy()

		centers = []
		scales = []

		anns = gt.loadAnns(gt.getAnnIds(imgIds=[img_id]))
		scores = []
		if args.use_gt:
			for box in [a["bbox"] for a in anns]:
				center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1], args.use_gt)
				centers.append(center)
				scales.append(scale)
		else:
			pred_boxes = box_model(img)
			scores = pred_boxes['instances'].scores.detach().cpu().numpy()
			pred_boxes = pred_boxes['instances'].pred_boxes
			for box in pred_boxes:
				center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1], args.use_gt)
				centers.append(center)
				scales.append(scale)

		# no bboxes found on image
		if len(scales) == 0:
			continue

		pose_preds = get_pose_estimation_prediction(pose_model, image_pose, centers, scales, transform=pose_transform)


		# N x 21 x 2
		keypoints = np.asarray([a["keypoints"] for a in anns]).reshape((-1, 21, 3))
		visibility = keypoints[:, :, 2]
		keypoints = keypoints[:, :, :2]


		areas = []
		if args.use_gt:
			areas = [ann["area"] for ann in anns]
		else:

			gt_bboxes = [x["bbox"] for x in anns]
			gt_bboxes = np.vstack(gt_bboxes)

			dt_bboxes = []
			for box in pred_boxes:
				dt_bboxes.append([box[0], box[1], box[2], box[3]])
			dt_bboxes = np.vstack(dt_bboxes)

			idxs_true, idxs_pred = match_bboxes(gt_bboxes, dt_bboxes, scores)

			if len(idxs_pred) >= 1:

				# find matched boxes + corresponding annotations
				gt_bboxes = [gt_bboxes[xi] for xi in idxs_true]
				dt_bboxes = [dt_bboxes[xi] for xi in idxs_pred]

				pose_preds = np.stack([pose_preds[xi, :, :] for xi in idxs_pred])
				keypoints = np.stack([keypoints[xi, :, :] for xi in idxs_true]) 

				areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in dt_bboxes]


		for ann_idx, area in enumerate(areas):
			idx = None
			if area < FIRST:
				idx = 0
			elif area < SECOND:
				idx = 1
			elif area < THIRD:
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
					for key, value in attrs[img_name].items():
						keypoint_accuracy_cats[key][value][ii] += 1
				elif not args.strict and (ii + 1 in acc):
					keypoint_accuracy[ii] += 1
					for key, value in attrs[img_name].items():
						keypoint_accuracy_cats[key][value][ii] += 1
				else:
					pose_preds[ann_idx, ii, :] = 0
					keypoints[ann_idx, ii, :] = 0
					num_removed += 1

				if ii + 1 in keypoint_present:
					keypoint_count[ii] += 1
					for key, value in attrs[img_name].items():
						keypoint_count_cats[key][value][ii] += 1


		visibility[visibility >= 1] = 1
		total_keypoints += np.sum(visibility)

		bbox = []
		distances = []

		if args.use_gt: 
			distances = np.sqrt(np.sum((keypoints - pose_preds) ** 2, axis=2))
			bbox = [max(a["bbox"][2], a["bbox"][3]) for a in anns]

		else:
			# gt_bboxes = [x["bbox"] for x in anns]
			# gt_bboxes = np.vstack(gt_bboxes)

			# dt_bboxes = []
			# for box in pred_boxes:
			# 	dt_bboxes.append([box[0], box[1], box[2], box[3]])
			# dt_bboxes = np.vstack(dt_bboxes)

			# idxs_true, idxs_pred = score_utils.match_bboxes(gt_bboxes, dt_bboxes, scores)

			if len(idxs_pred) >= 1:

				# # find matched boxes + corresponding annotations
				# gt_bboxes = [gt_bboxes[xi] for xi in idxs_true]
				# dt_bboxes = [dt_bboxes[xi] for xi in idxs_pred]

				# pose_preds = np.stack([pose_preds[xi, :, :] for xi in idxs_pred])
				# keypoints = np.stack([keypoints[xi, :, :] for xi in idxs_true]) 

				distances = np.sqrt(np.sum((keypoints - pose_preds) ** 2, axis=2))

				bbox = [max(a[2], a[3]) for a in gt_bboxes]

			# unmatched keypoints dist is inf
			if len(idxs_pred) < len(anns):
				filler = np.zeros((len(anns) - len(idxs_pred), 21))
				filler.fill(np.inf)

				if len(idxs_pred) >= 1:
					distances = np.vstack([distances, filler])
				else:
					distances = filler

				bbox.extend([0 for i in range(len(anns) - len(idxs_pred))])

		# distances[visibility < 1] = np.nan

		for key, value in attrs[img_name].items():
			pck_distances[key][value].append(distances)
			pck_bbox_norms[key][value].extend(bbox)


	with open("pck_category_analysis_filtered_" + ("strict_" if args.strict else "soft_") + ("gt" if args.use_gt else "dt") + ("_test" if args.test else "") + ".txt", "w") as file:

		for attribute, values in pck_distances.items():
			for value_name in values:

				print(attribute.upper(), value_name.upper(), file=file)

				distances = np.vstack(pck_distances[attribute][value_name])
				gt_reference = np.stack([np.asarray(pck_bbox_norms[attribute][value_name])] * 21, axis=-1)

				pck = np.nanmean(distances <= args.alpha * gt_reference, axis=0)


				print("PCK@" + str(args.alpha)+ " per keypoint", pck, file=file)
				print("Avg PCK" + str(args.alpha), np.nanmean(distances <= args.alpha * gt_reference), file=file)
				print("PCK@" + str(args.alpha)+ " Thumb", np.nanmean(distances[:, 1:5] <= args.alpha * gt_reference[:, 1:5]), file=file)
				print("PCK@" + str(args.alpha)+ " Pointer", np.nanmean(distances[:, 5:9] <= args.alpha * gt_reference[:, 5:9]), file=file)
				print("PCK@" + str(args.alpha)+ " Middle", np.nanmean(distances[:, 9:13] <= args.alpha * gt_reference[:, 9:13]), file=file)
				print("PCK@" + str(args.alpha)+ " Ring", np.nanmean(distances[:, 13:17] <= args.alpha * gt_reference[:, 13:17]), file=file)
				print("PCK@" + str(args.alpha)+ " Pinky", np.nanmean(distances[:, 17:21] <= args.alpha * gt_reference[:, 17:21]), file=file)
				print("PCK@" + str(args.alpha)+ " Wrist", np.nanmean(distances[:, 0] <= args.alpha * gt_reference[:, 0]), file=file)

				key_acc = keypoint_accuracy_cats[attribute][value_name]
				key_count = keypoint_count_cats[attribute][value_name]
				print("Keypoint Accuracy per keypoint", key_acc / key_count, file=file)
				print("Avg Keypoint Accuracy", np.sum(key_acc) / np.sum(key_count), file=file)
				print("Avg Keypoint Accuracy Thumb", np.sum(key_acc[1:5]) / np.sum(key_count[1:5]), file=file)
				print("Avg Keypoint Accuracy Pointer", np.sum(key_acc[5:9]) / np.sum(key_count[5:9]), file=file)
				print("Avg Keypoint Accuracy Middle", np.sum(key_acc[9:13]) / np.sum(key_count[9:13]), file=file)
				print("Avg Keypoint Accuracy Ring", np.sum(key_acc[13:17]) / np.sum(key_count[13:17]), file=file)
				print("Avg Keypoint Accuracy Pinky", np.sum(key_acc[17:21]) / np.sum(key_count[17:21]), file=file)



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



if __name__ == '__main__':
	main()
