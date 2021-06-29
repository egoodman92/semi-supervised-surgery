from typing import Dict, Iterable, Union

from collections import defaultdict
import contextlib
from copy import deepcopy
import io

import numpy as np
import pandas as pd

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class KeypointHandler:
    """Keypoint evaluation utility.

    For keypoint evaluation, there should only be one category id for both ground truth and detection annotations.

    Predicted annotations are matched greedily (based on bounding box IOU) with ground truth annotations.
    Distances are computed between matched prediction and ground truth keypoint annotations.

    By default, ground truth annotations with `'ignore'` flag set to 1 or those missing ground truth
    keypoints are not counted. As a result, you may see fewer ground truth annotations than those annotated.
    """

    def __init__(self, gt_coco, dt_coco):
        self.gt_coco = gt_coco
        self.dt_coco = dt_coco

        assert gt_coco.getCatIds() == dt_coco.getCatIds(), "gt cats: {}\ndt cats: {}".format(gt_coco.getCatIds(),
                                                                                             dt_coco.getCatIds())
        assert len(gt_coco.getCatIds()) == 1, "Expected one category, got {}".format(len(gt_coco.getCatIds()))

        self._prepare()

    def _prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval = COCOeval(self.gt_coco, self.dt_coco, iouType="bbox")
            coco_eval.params.areaRng = [[0, 10000000000.0]]
            coco_eval.params.iouThrs = [0.]
            coco_eval.evaluate()

        eval_imgs = {(x["image_id"], x["category_id"]): x for x in coco_eval.evalImgs if x is not None}
        self._mapping = self._init_mapping(eval_imgs.values())

        return coco_eval

    def _init_mapping(self, eval_imgs):
        """
        Creates dictionary of gt_ids default ids and
        N: # ground truth detection instances
        K: # keypoints
        """
        gt_coco = self.gt_coco
        dt_coco = self.dt_coco

        mappings = defaultdict(list)

        for img_data in eval_imgs:
            valid_gt = ~(img_data["gtIgnore"].astype(np.bool))
            gt_ids = np.array(img_data["gtIds"])[valid_gt]  # skip over gt annotations that we should ignore
            gt_matches = (img_data["gtMatches"][:, valid_gt]).flatten()

            # Loop over pairs of ground truth annotation ids and matched detection annotation ids
            for gt_id, dt_id in zip(list(gt_ids), list(gt_matches)):
                gt_ann = gt_coco.loadAnns(int(gt_id))[0]['keypoints']
                if dt_id:
                    dt_ann = dt_coco.loadAnns(int(dt_id))[0]['keypoints']
                else:
                    # If no detection is made, then the distances are infinite.
                    dt_ann = np.inf * np.ones(len(gt_ann))
                    dt_ann[2::3] = 0

                # Do not compute distances over gt annotations that do not have keypoints.
                if not gt_ann:
                    continue

                gt_x, gt_y, gt_v = gt_ann[::3], gt_ann[1::3], gt_ann[2::3]
                dt_x, dt_y, dt_c = dt_ann[::3], dt_ann[1::3], dt_ann[2::3]

                # Distance, visibility, and confidence for each keypoint.
                distance = np.sqrt(
                    (np.asarray(gt_x) - np.asarray(dt_x)) ** 2 + (np.asarray(gt_x) - np.asarray(dt_x)) ** 2)
                visibility = np.asarray(gt_v)
                confidence = np.asarray(dt_c)

                info = {
                    "gt_id": gt_id,  # shape (N,)
                    "dt_id": dt_id,  # shape (N, )
                    "distance": distance,  # shape (N, K)
                    "visibility": visibility,  # shape (N, K)
                    "confidence": confidence,  # shape (N, K)
                }
                for k, v in info.items():
                    mappings[k].append(v)

        for k, v in mappings.items():
            mappings[k] = np.asarray(v)

        return mappings

    def num_instances(self):
        return len(self.mappings["gt_id"])

    @property
    def mapping(self):
        return deepcopy(self._mapping)

    def __str__(self):
        s = "Num instances: {}\n".format(len(self._mapping["gt_id"]))
        s += "Num detections: {}\n".format(sum(self._mapping["dt_id"] != 0))
        s += "Num keypoints: {}\n".format(self._mapping["visibility"].shape[1])
        return s

    def _format_reference(self, reference, distance, num_keypoints, mappings, keypoint_names):
        """Computes and formats ground truth references into array of shape (N, K).
        
        N: Number of instances
        K: Number of keypoints

        Returns:
            np.ndarray: Shape `(N, K)`.
        """
        if isinstance(reference, (int, float)):
            gt_reference = reference * np.ones(len(self.gt_coco.loadAnns(mappings["gt_id"])))
        else:
            if reference == "torso_diameter" and "torso_diameter" not in self.gt_coco.loadAnns(mappings["gt_id"])[0]:
                kp_names = {v: k for k, v in keypoint_names.items()}
                lh_idx = kp_names["left_hip"] - 1
                rs_idx = kp_names["right_shoulder"] - 1

                gt_reference = []
                for ann in self.gt_coco.loadAnns(mappings["gt_id"]):
                    if "keypoints" not in ann or not ann["keypoints"]:
                        gt_reference.append(float("nan"))
                        continue

                    kps = ann["keypoints"]
                    x, y, v = kps[::3], kps[1::3], kps[2::3]
                    if v[lh_idx] == 0 or v[rs_idx] == 0:
                        gt_reference.append(float("nan"))
                        continue

                    dist = np.sqrt((x[lh_idx] - x[rs_idx])**2 + (y[lh_idx] - y[rs_idx])**2)
                    gt_reference.append(dist)
            else:
                try:
                    gt_reference = [ann[reference] for ann in self.gt_coco.loadAnns(mappings["gt_id"])]
                except KeyError as _:
                    raise KeyError("reference {} not found in annotation file".format(reference))

                if reference == "bbox":
                    gt_reference = [max(x[2], x[3]) for x in gt_reference]
                assert all([isinstance(x, (int, float)) for x in gt_reference])

        gt_reference = np.stack([np.asarray(gt_reference)] * num_keypoints, axis=-1)  # shape (N, K)
        return gt_reference

    def compute_pck(self,
                    alphas: Iterable[float] = np.linspace(0, 1, 11),
                    reference: Union[str, float] = "bbox",
                    norm_factor: float = 1.0,
                    vis_threshold=1,
                    keypoint_names: Dict[int, str] = None,
                    add_average: bool = False,
                    ):
        """Compute PCK per keypoint.

        PCK is calculated with respect to some reference (bbox size, head size, etc.)
        Use the `reference` argument to specify the annotation field used as the reference.

        Distances are measured as :math:`dist <= alpha * norm_factor * size(reference)`
        where the `size()` operation is calcuated based on the reference type.

        NOTE: If you change any defaults here, make sure to update the defaults in evaluation/evaluator.py

        Args:
            alphas (:obj:`Iterable[float]`): An iterable of floats from 0-1.
            reference (:obj:`str` or :obj:`int/float`): Reference type. String options are `'bbox'`, `'head_size'`.
                If number, will be considered as pixel distance.
            norm_factor (float): Normalization factor.
            vis_threshold (int): Minimum COCO visibility to count. Keypoints with visibility
                less thant `vis_threshold` will be discarded in computation.
            keypoint_names (Dict[int, str]): Map from keypoint id to name.
            add_average (`bool`, optional): Add `"mean"` column in returned dataframe. 
        Returns:
            pd.DataFrame: DataFrame of PCK per keypoint for varying alphas.
        """
        mappings = self.mapping
        distance: np.ndarray = mappings["distance"]
        visibility = mappings["visibility"]
        # distance[visibility < vis_threshold] = np.nan

        num_keypoints = distance.shape[1]
        gt_reference = self._format_reference(reference, distance, num_keypoints, mappings, keypoint_names)

        pcks = []
        for alpha in alphas:
            pcks.append(np.nanmean(distance <= alpha * norm_factor * gt_reference, axis=0))

        df = pd.DataFrame(pcks,
                          columns=[keypoint_names[i + 1] if keypoint_names else "kp_{}".format(i + 1) for i in
                                   range(num_keypoints)],
                          )
        if add_average:
            df["mean"] = df.mean(axis=1)

        df["alpha"] = alphas

        return df
