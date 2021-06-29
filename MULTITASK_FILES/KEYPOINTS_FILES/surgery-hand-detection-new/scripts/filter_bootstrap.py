from __future__ import division
import scipy.optimize
import numpy as np
import json
import re
import cv2
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


def cull_annotations():
	gt = json.load(open("train_validated_keypoints_only.json", "r"))
	boot = json.load(open("hand_keypoints_train_bootstrap.json", "r"))

	# for each image in GT, find entries in boot
	gt_images = gt["images"]
	gt_anns = {}
	boot_anns = {}
	boot_imgs = {}
	max_img_id = 0
	max_ann_id = 0

	for img in gt_images:
		img_id = img["id"]
		# if img_id > max_img_id:
		# 	max_img_id = img_id

		img["id"] = max_img_id + 1
		gt_anns[img["file_name"]] = []

		for ann in gt["annotations"]:
			# if ann["id"] > max_ann_id:
			# 	max_ann_id = ann["id"] 

			if ann["image_id"] == img_id:
				ann["image_id"] = img["id"]
				ann["id"] = max_ann_id
				gt_anns[img["file_name"]].append(ann)

				max_ann_id += 1

		max_img_id += 1


	for img in boot["images"]:
		img_id = img["id"]
		boot_imgs[img["file_name"]] = img
		boot_anns[img["file_name"]] = []

		for ann in boot["annotations"]:
			if ann["image_id"] == img_id:
				boot_anns[img["file_name"]].append(ann)


	boot_new_images = []
	count = 0
	total = 0
	imgs_to_remove = set()
	for gt_img in gt_images:
		img_name = gt_img["file_name"]

		vid_id = re.sub("-\d{9}.jpg", "", img_name)
		frame = int(re.search("\d{9}", img_name).group())

		boot_frames = set(["{vid_id}-{frame:09d}.jpg".format(vid_id=vid_id, frame=frame+i) for i in range(-5, 6)])
		boot_frames.remove(img_name)

		# for each entry in boot, match bboxes (x, y, w, h)
		gt_bboxes = []
		for x in gt_anns[img_name]:
			gt_bboxes.append([x["bbox"][0], x["bbox"][1], x["bbox"][0] + x["bbox"][2], x["bbox"][1] + x["bbox"][3]])
		gt_bboxes = np.array(gt_bboxes)


		for f in boot_frames:

			boxes = []
			if f not in boot_anns:
				continue

			# reassign image
			boot_imgs[f]["id"] = max_img_id + 1

			for x in boot_anns[f]:
				boxes.append([x["bbox"][0], x["bbox"][1], x["bbox"][0] + x["bbox"][2], x["bbox"][1] + x["bbox"][3]])
			boxes = np.array(boxes)

			idx_true, idxs_pred, ious, labels = match_bboxes(gt_bboxes, boxes)

			
			total += 1
			if len(idxs_pred) >= 1:
				count += 1

				# image_debug = cv2.imread("./bootstrap/" + f)

				# find matched boxes + corresponding annotations
				gt_using = [gt_anns[img_name][xi] for xi in idx_true]
				boot_using = [boot_anns[f][xi] for xi in idxs_pred]

				# eliminate keypoints
				remove_idxs = []
				using = []
				for b in range(len(boot_using)):
					gt_ann = gt_using[b]
					boot_ann = boot_using[b]

					boot_ann["image_id"] = max_img_id + 1
					boot_ann["id"] = max_ann_id + 1

					keypoints = boot_ann["keypoints"]

					for i in range(21):
						if gt_ann["keypoints"][i * 3 + 2] == 0:
							keypoints[i * 3] = 0
							keypoints[i * 3 + 1] = 0
							keypoints[i * 3 + 2] = 0
						# else:
						# 	cv2.circle(image_debug, (int(keypoints[i * 3]), int(keypoints[i * 3 + 1])), 4, keypoint_colors[str(i + 1)], -1)
						# 	cv2.putText(image_debug, str(i + 1), (int(keypoints[i * 3] - 4), int(keypoints[i * 3 + 1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

					if sum(keypoints) == 0:
						remove_idxs.append(b)
						continue
					using.append(boot_ann)

					boot_ann["keypoints"] = keypoints
					
					box = boot_ann["bbox"]
					# cv2.rectangle(image_debug, (int(box[0]), int(box[1])), (int(box[2] + box[0]) , int(box[3] + box[1])), color=(0, 255, 0),thickness=3) 
					# cv2.rectangle(image_debug, (int(box[0]), int(box[1])), (int(box[2]) , int(box[3])), color=(0, 255, 0),thickness=3) 

					# fix error bbox coco error here # Error is in train _bootstrap
					boot_ann["bbox"] = [box[0], box[1], box[2] - box[0], box[3] - box[1]] # x y w h
					boot_ann["area"] = (box[2] - box[0]) * (box[3] - box[1])

					max_ann_id += 1


				# remove image if no more annotations
				if len(remove_idxs) == len(boot_using):
					imgs_to_remove.add(f)

				# boot_using = [x for i, x in enumerate(boot_using) if i not in remove_idxs]

				# add to new annotations
				boot_new_images.extend(using)

				# save image for debugging
				# cv2.imwrite("./debug_bootstrap_train_filter/debug_" + f, image_debug)
			else:	
				# eliminate image 
				imgs_to_remove.add(f)

				print(f)

			max_img_id += 1

	print(len(boot_new_images))
	boot["annotations"] = boot_new_images
	for v in gt_anns.values():
		boot["annotations"].extend(v)
	print(len(boot["annotations"]))

	boot["images"] = list(boot_imgs.values())
	boot["images"].extend(gt["images"])
	print(len(boot["images"]), len(imgs_to_remove), len(gt["images"]))

	removed = set()
	imgs_cpy = list(boot["images"])
	for img in imgs_cpy:
		if img["file_name"] in imgs_to_remove:
			boot["images"].remove(img)
			removed.add(img["id"])
			if img in gt["images"]:
				print("overlap")

	print(len(boot["images"]), len(boot["annotations"]))
	# remove annotations on removed images
	cpy = list(boot["annotations"])
	for ann in cpy:
		if ann["image_id"] in removed:
			boot["annotations"].remove(ann)
			print("hi")








	json.dump(boot, open("train_boostrap_filtered_validated_2.json", "w"))


	print(count, total)



	# get rid of unmatched boxes

	# get rid of keypoints that aren't labled, occluded, w/in a certain distance

	# write to image so you can see what they now look like

cull_annotations()
	