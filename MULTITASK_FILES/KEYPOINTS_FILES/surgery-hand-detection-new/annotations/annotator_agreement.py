from __future__ import division
import json
import numpy as np
from PIL import Image
from collections import defaultdict
import scipy.optimize
import re
import cv2


def bbox_iou(boxA, boxB):
	# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
	# ^^ corrected.
	
	# Determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	interW = xB - xA + 1
	interH = yB - yA + 1

	# Correction: reject non-overlapping boxes
	if interW <=0 or interH <=0 :
		return -1.0

	interArea = interW * interH
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou



def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.05):
	'''
	Given sets of true and predicted bounding-boxes,
	determine the best possible match.
	Parameters
	----------
	bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
	  The number of bboxes, N1 and N2, need not be the same.
	
	Returns
	-------
	(idxs_true, idxs_pred, ious, labels)
		idxs_true, idxs_pred : indices into gt and pred for matches
		ious : corresponding IOU value of each match
		labels: vector of 0/1 values for the list of detections
	'''
	n_true = bbox_gt.shape[0]
	n_pred = bbox_pred.shape[0]
	MAX_DIST = 1.0
	MIN_IOU = 0.0

	# NUM_GT x NUM_PRED
	iou_matrix = np.zeros((n_true, n_pred))
	for i in range(n_true):
		for j in range(n_pred):
			iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])

	if n_pred > n_true:
	  # there are more predictions than ground-truth - add dummy rows
	  diff = n_pred - n_true
	  iou_matrix = np.concatenate( (iou_matrix, 
									np.full((diff, n_pred), MIN_IOU)), 
								  axis=0)

	if n_true > n_pred:
	  # more ground-truth than predictions - add dummy columns
	  diff = n_true - n_pred
	  iou_matrix = np.concatenate( (iou_matrix, 
									np.full((n_true, diff), MIN_IOU)), 
								  axis=1)

	# call the Hungarian matching
	idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)


	if (not idxs_true.size) or (not idxs_pred.size):
		ious = np.array([])
	else:
		ious = iou_matrix[idxs_true, idxs_pred]

	# remove dummy assignments
	sel_pred = idxs_pred<n_pred
	idx_pred_actual = idxs_pred[sel_pred] 
	idx_gt_actual = idxs_true[sel_pred]
	ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
	sel_valid = (ious_actual > IOU_THRESH)
	label = sel_valid.astype(int)

	return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label 

def compile_bboxes(lst, height, width):
	bboxes = []
	for label in lst:
		# x1 y1 x2 y2
		box = [float(label["bounding_box_position"]['left']) * width,
			   float(label["bounding_box_position"]['top']) * height,
			   (float(label["bounding_box_position"]['width']) + float(label["bounding_box_position"]['left']))* width,
			   (float(label["bounding_box_position"]['height']) + float(label["bounding_box_position"]['top'])) * height]
		bboxes.append(box)

	if len(bboxes) > 0:
		return np.vstack(bboxes)
	return None

def format_keys(d):
	if d is None:
		return d

	ret = [None for i in range(21)]
	if type(d) == dict:

		for k, v in d.items():
			ret[int(k)] = v

		return ret

	else:
		if len(d) < 21:
			d.extend([None for i in range(21 - len(d))])
		return d



rohan = json.load(open("tools second pass/rohan_data.json", "r"))['data'] 
stephen = json.load(open("tools second pass/stephen_data.json", "r"))['data']

validated = json.load(open("validated/marvl-surgery-annotator-validate-export.json", "r"))
del validated['index']

validated_annotations = []

for val_lst in validated.values():
	validated_annotations.extend(val_lst['data'])

original_annotations = []

for elem in rohan:
	if elem['object_type'] == 'image':
		if elem['id'] < 1716:
			continue
		original_annotations.append(elem)

for elem in stephen:
	if elem['object_type'] == 'image':
		if elem['id'] >= 1716:
			continue
		original_annotations.append(elem)


# map name -> annotations
orig_img_data = {}
val_img_data = {}

for elem in original_annotations:
	if elem['object_type'] == 'image':
		orig_img_data[elem['name']] = elem

for elem in validated_annotations:
	if elem['object_type'] == 'image':
		val_img_data[elem['name']] = elem


new_bboxes = 0
deleted_bboxes = 0
iou_total = 0
total_matched = 0
iou_changed_total = 0 
num_changed = 0
unchanged_boxes = 0

new_keypoints = [0 for i in range(21)]
deleted_keypoints = [0 for i in range(21)]
avg_dist = [0 for i in range(21)]
keypoint_count = [0 for i in range(21)]
keypoints_changed = [0 for i in range(21)]
keypoints_changed_dists = []

# new_tools = defaultdict(int)
# deleted_tools = defaultdict(int)
# iou_total_tools = defaultdict(float)
# total_matched_tools = defaultdict(float)
# iou_changed_total_tools = defaultdict(float)
# num_changed_tools = defaultdict(float)
# unchanged_boxes_tools = defaultdict(float)

# for each image
for file_name, orig_anns in orig_img_data.items():
	img = Image.open('../all_images/' + file_name)
	width, height = img.size

	if 'hand_labels' in orig_anns and 'hand_labels' in val_img_data[file_name]:
		orig_hands = compile_bboxes(orig_anns['hand_labels'], height, width)
		ann_hands = compile_bboxes(val_img_data[file_name]['hand_labels'], height, width)

		if orig_hands is not None and ann_hands is not None:
			gt_idxs, dt_idxs, ious, labels = match_bboxes(orig_hands, ann_hands)

			iou_total += np.sum(ious)
			total_matched += ious.shape[0]

			changed = ious[ious < 1.0]
			iou_changed_total += np.sum(changed)
			num_changed += changed.shape[0]

			unchanged_boxes += ious.shape[0] - changed.shape[0]

			deleted_bboxes += orig_hands.shape[0] - gt_idxs.shape[0]
			new_bboxes += ann_hands.shape[0] - dt_idxs.shape[0]

			# should not occur
			if len(gt_idxs) != len(set(gt_idxs)):
				print('duplicates')


			o_hands = [orig_anns['hand_labels'][x] for x in gt_idxs]
			v_hands = [val_img_data[file_name]['hand_labels'][x] for x in dt_idxs]

			# distance between v1 keypoints and v2 keypoints
			for hi in range(len(o_hands)):

				o_keys = format_keys(o_hands[hi].get('keypoints', None))
				v_keys = format_keys(v_hands[hi].get('keypoints', None))

				if o_keys is None and v_keys is None:
					continue
				elif o_keys is None:
					# all keypoints are new
					for ki in range(21): 
						if v_keys[ki] is not None:
							new_keypoints[ki] += 1

				elif v_keys is None:
					# all keypoints were deleted
					for ki in range(21):
						if o_keys[ki] is not None:
							deleted_keypoints[ki] += 1
				else:

					for ki in range(21):
						# if both keypts exist, calc dist
						if o_keys[ki] is not None and v_keys[ki] is not None:
							x1 = width * float(o_keys[ki]['position']['left'])
							y1 = height * float(o_keys[ki]['position']['top'])
							x2 = width * float(v_keys[ki]['position']['left'])
							y2 = height * float(v_keys[ki]['position']['top'])
							dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
							avg_dist[ki] += dist
							if dist > 0:
								keypoints_changed[ki] += 1
								keypoints_changed_dists.append(dist)
							keypoint_count[ki] += 1

						# new keypoint
						elif v_keys[ki] is not None:
							new_keypoints[ki] += 1

						# deleted keypoint
						elif o_keys[ki] is not None: 
							deleted_keypoints[ki] += 1

			# unmatched bboxes
			unmatched_orig = [lab for x, lab in enumerate(orig_anns['hand_labels']) if x not in gt_idxs]
			unmatched_val = [lab for x, lab in enumerate(val_img_data[file_name]['hand_labels']) if x not in dt_idxs]

			# new keypoints
			for hand in unmatched_val:
				keys = format_keys(hand.get('keypoints', None))
				if keys is None:
					continue

				for ki in range(21):
					if keys[ki] is not None:
						new_keypoints[ki] += 1

			# # deleted keypoints
			for hand in unmatched_orig:
				keys = format_keys(hand.get('keypoints', None))
				if keys is None:
					continue

				for ki in range(21):
					if keys[ki] is not None:
						deleted_keypoints[ki] += 1

		elif orig_hands is not None:
			deleted_bboxes += len(orig_anns['hand_labels'])
			for hand in orig_anns['hand_labels']:
				keys = format_keys(hand.get('keypoints', None))
				if keys is None:
					continue
				for ki in range(21):
					if hand['keypoints'][ki] is not None:
						deleted_keypoints[ki] += 1

		elif ann_hands is not None:
			new_bboxes += len(val_img_data[file_name]['hand_labels'])
			for hand in val_img_data[file_name]['hand_labels']:
				keys = format_keys(hand.get('keypoints', None))
				if keys is None:
					continue
				for ki in range(21):
					if keys[ki] is not None:
						new_keypoints[ki] += 1

	elif 'hand_labels' in orig_anns:
		deleted_bboxes += len(orig_anns['hand_labels'])
		for hand in orig_anns['hand_labels']:
			keys = format_keys(hand.get('keypoints', None))
			if keys is None:
				continue
			for ki in range(21):
				if hand['keypoints'][ki] is not None:
					deleted_keypoints[ki] += 1

	elif 'hand_labels' in val_img_data[file_name]:
		new_bboxes += len(val_img_data[file_name]['hand_labels'])
		for hand in val_img_data[file_name]['hand_labels']:
			keys = format_keys(hand.get('keypoints', None))
			if keys is None:
				continue
			for ki in range(21):
				if keys[ki] is not None:
					new_keypoints[ki] += 1

	# if 'tool_labels' in orig_anns and 'tool_labels' in val_img_data[file_name]:
	# 	orig_tools = compile_bboxes(orig_anns['tool_labels'], height, width)
	# 	ann_tools = compile_bboxes(val_img_data[file_name]['tool_labels'], height, width)

	# 	if orig_tools is not None and ann_tools is not None:
	# 		gt_idxs, dt_idxs, ious, labels = match_bboxes(orig_tools, ann_tools)

	# 	elif orig_tools is not None:
	# 		for ann in orig_anns['tool_labels']:
	# 			deleted_tools[ann['category']] += 1
	# 	elif ann_tools is not None:
	# 		for ann in val_img_data[file_name]['tool_labels']:
	# 			new_tools[ann['category']] += 1

	# elif 'tool_labels' in orig_anns:
	# 	for ann in orig_anns['tool_labels']:
	# 		deleted_tools[ann['category']] += 1

	# elif 'tool_labels' in val_img_data[file_name]:
	# 	for ann in val_img_data[file_name]['tool_labels']:
	# 		new_tools[ann['category']] += 1



print("new bboxes ", new_bboxes)
print("deleted bboxes ", deleted_bboxes)
print("avg iou total ", iou_total / total_matched)
print("avg iou of changed bbox ", iou_changed_total / num_changed)
print("avg keypoint distance from original ", np.asarray(avg_dist) / np.asarray(keypoint_count))
print("avg keypoint distance from original (changed keypoints) ", np.asarray(avg_dist) / np.asarray(keypoints_changed))
print("changed bounding boxes ", num_changed)
print("unchanged bounding boxes ", unchanged_boxes)
print("new keypoints ", new_keypoints)
print("deleted keypoints ", deleted_keypoints)
print("keypoints changed ", keypoints_changed)
print("total keypoints changed ", sum(keypoints_changed))
print("percent distances over 10", sum(filter(lambda x: True if x > 10 else False, keypoints_changed_dists)) / len(keypoints_changed_dists))

print(sum(keypoint_count))


