# based off of https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4

from __future__ import division
import scipy.optimize
import numpy as np
import json
import re
import cv2

def computeIOUs(gt, dt):
	# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
	
	# Determine the (x, y)-coordinates of the intersection rectangle
	xA = max(gt[0], dt[0])
	yA = max(gt[1], dt[1])
	xB = min(gt[2] + gt[0], dt[2])
	yB = min(gt[3] + gt[1], dt[3])

	interW = xB - xA
	interH = yB - yA

	if interW <=0 or interH <=0 :
		return -1.0

	interArea = interW * interH
	gtArea = gt[2] * gt[3]
	dtArea = (dt[2] - dt[0]) * (dt[3] - dt[1])
	iou = interArea / float(gtArea + dtArea - interArea)
	return iou



def match_bboxes(bbox_gt, bbox_pred, scores, IOU_THRESH=0.5):
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
	'''
	n_true = bbox_gt.shape[0]
	n_pred = bbox_pred.shape[0]
	MAX_DIST = 1.0
	MIN_IOU = 0.0

	# sort predictions by score
	score_idxs = np.argsort(-1 * scores)
	bbox_pred = bbox_pred[score_idxs]


	# NUM_GT x NUM_PRED
	iou_matrix = np.zeros((n_true, n_pred))
	for i in range(n_true):
		for j in range(n_pred):
			iou_matrix[i, j] = computeIOUs(bbox_gt[i,:], bbox_pred[j,:])


	# heavily inspired by COCOEvaluator
	idxs_true = []
	idxs_pred = []
	for dind, d in enumerate(bbox_pred):

		m = -1 
		iou = IOU_THRESH

		for gind, g in enumerate(bbox_gt):

			# continue to next gt unless better match made
			if iou_matrix[gind, dind] < iou:
				continue

			iou = iou_matrix[gind, dind]
			m = gind

			break

		if m > -1:
			idxs_true.append(m)
			idxs_pred.append(dind)

	
	return np.asarray(idxs_true), np.asarray(idxs_pred)
