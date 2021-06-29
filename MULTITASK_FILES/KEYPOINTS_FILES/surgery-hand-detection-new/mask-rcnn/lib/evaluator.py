"""Utilities for evaluating hand datasets."""
import logging

import copy
import itertools
import json
import os
import time

from collections import OrderedDict
import numpy as np
import pandas as pd

from fvcore.common.file_io import PathManager
from pycocotools.cocoeval import COCOeval as _COCOeval
import torch

from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator as _COCOEvaluator
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco
import detectron2.utils.comm as comm

from .util import KeypointHandler

class COCOeval(_COCOeval):
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


class CobraCOCOEvaluator(_COCOEvaluator):
    """Evaluate instance keypoint results for hands using COCO-like API.
    
    Because hand datasets do not have well defined OKS parameters, different
    methods have to be used for keypoint detection.

    In this case, we leverage two other techniques for evaluating keypoints:
        - F-score
        - PCK

    Both are conditioned on some alpha.

    Args:
    dataset_name (str): name of the dataset to be evaluated.
        It must have either the following corresponding metadata:

            "json_file": the path to the COCO format annotation

        Or it must be in detectron2's standard dataset format
        so it can be converted to COCO format automatically.
    cfg (CfgNode): config instance
    distributed (True): if True, will collect results from all ranks for evaluation.
        Otherwise, will evaluate the results in the current process.
    output_dir (str): optional, an output directory to dump all
        results predicted on the dataset. The dump contains two files:

        1. "instance_predictions.pth" a file in torch serialization
           format that contains all the raw original predictions.
        2. "coco_instances_results.json" a json file in COCO's result
           format.
    load (bool): Load data from :var:`output_dir`. Defaults to `False`.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir: str=None, load: bool=False, iteration=""):
        super().__init__(dataset_name, cfg, distributed, output_dir=output_dir)
        self._load = load
        self._alphas = cfg.TEST.KEYPOINT_ALPHAS
        self._iteration = iteration

    def evaluate(self):
        file_path = os.path.join(self._output_dir, "instances_predictions.pth")
        if self._load:
            with PathManager.open(file_path, "rb") as f:
                self._predictions = torch.load(f)
        else:
            if self._distributed:
                comm.synchronize()
                self._predictions = comm.gather(self._predictions, dst=0)
                self._predictions = list(itertools.chain(*self._predictions))

                if not comm.is_main_process():
                    return {}

            if len(self._predictions) == 0:
                self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
                return {}

            if self._output_dir:
                PathManager.mkdirs(self._output_dir)
                with PathManager.open(file_path, "wb") as f:
                    torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "proposals" in self._predictions[0]:
            self._eval_box_proposals()
        if "instances" in self._predictions[0]:
            self._eval_predictions(set(self._tasks))

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks):
        """
        Evaluate self._predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            filename = "iter={}_coco-results.json".format(self._iteration) if self._iteration else "coco-results.json"
            file_path = os.path.join(self._output_dir, filename)
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            if task == 'keypoints':
                self._eval_predictions_keypoints()
            else:
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, task
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )

                res = self._derive_coco_results(
                    coco_eval, task, class_names=self._metadata.get("thing_classes")
                )
                self._results[task] = res

    def _eval_predictions_keypoints(self):
        coco_gt = self._coco_api
        coco_dt = coco_gt.loadRes(self._coco_results)

        # PCK analysis
        # TODO: Add option for logging all/specific keypoints to tensorboard instead of the mean.
        pck_params = self._metadata.keypoint_pck_params
        pck_params["alphas"] = self._alphas
        pck_params["keypoint_names"] = {i+1: x for i, x in enumerate(self._metadata.keypoint_names)}
        kp_handler = KeypointHandler(coco_gt, coco_dt)
        pck_df = kp_handler.compute_pck(**pck_params, add_average=True)
        self._results["keypoints"] = {"PCK@{}".format(alpha): pck_df[pck_df["alpha"] == alpha]["mean"].values[0]
                                          for alpha in self._alphas
                                     }

        #self._logger.info(pck_df)
