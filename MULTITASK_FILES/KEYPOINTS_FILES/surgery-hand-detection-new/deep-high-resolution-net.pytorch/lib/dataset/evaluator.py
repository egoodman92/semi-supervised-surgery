import logging

import copy
import itertools
import json
import os
import time

from collections import OrderedDict
import numpy as np
import pandas as pd

from pycocotools.cocoeval import COCOeval
import torch


class PCKCOCOeval(COCOeval):
    """Run COCO evaluation with percentage of correct keypoints (PCK) per image.

    Args:
        cocoGt: Ground truth data in COCO format.
        cocoDt: Detection data in COCO format.
        iouType (:obj:`str`, optional): IOU Type. One of ``'bbox'``, ``'segm'``, or ``'keypoints'``. 
            Defaults to ``'segm'``.
        alpha (:obj:`float`, optional): Alpha parameter for keypoint accuracy calculations (PCK, etc.).
            Defaults to ``0.1``.
    """
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm', alpha=0.1):
        super().__init__(cocoGt, cocoDt, iouType)
        self._alpha = alpha

    def evaluate(self):
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computePCK
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computePCK(self, imgId, catId):
        """Compute percent of correct keypoints (PCK) between ground truth and predicted instances in image.

        Args:
            imgId (int): Index of image to process.
            catId (int): Category id of instances to process.

        Returns:
            Array-like of IOUs indexed by ``[dt_ind, gt_ind]``.
        """
        p = self.params
        alpha = self._alpha

        # Dimension should be Nxm.
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        if len(gts) == 0 or len(dts) == 0:
            return []

        ious = np.zeros((len(dts), len(gts)))
        k = len(gts[0]['keypoints']) // 3  # number or keypoints

        # Compute  each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)

                e = np.sqrt(dx**2 + dy**2) <= alpha*np.max((bb[2], bb[3]))  # Within 10% of the bbox.
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(e) / e.shape[0]
        return ious
