from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
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

# import some common libraries
import numpy as np
import cv2
import os
import random

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import matplotlib.pyplot as plt
import plotly.graph_objects as go


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


def zero_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=(b != 0))

def parse_args():
    parser = argparse.ArgumentParser(description='PCK analysis')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=0.5)
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


    pcks = {'hand': [], 'thumb': [], 'pointer': [], 'middle': [], 'ring': [], 'pinky': []}
    vis = {'hand': [], 'thumb': [], 'pointer': [], 'middle': [], 'ring': [], 'pinky': []}
    occ = {'hand': [], 'thumb': [], 'pointer': [], 'middle': [], 'ring': [], 'pinky': []}
    lab = {'hand': [], 'thumb': [], 'pointer': [], 'middle': [], 'ring': [], 'pinky': []}
    for img_id in gt.getImgIds():

        path = "../deep-high-resolution-net.pytorch/data/coco/images/all_images/" + gt.loadImgs([img_id])[0]['file_name']
 
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

        # no bbox found on img
        if len(scales) == 0:
            continue

        pose_preds = get_pose_estimation_prediction(pose_model, image_pose, centers, scales, transform=pose_transform)


        # N x 21 x 2
        keypoints = np.asarray([a["keypoints"] for a in anns]).reshape((-1, 21, 3))
        visibility = keypoints[:, :, 2]
        keypoints = keypoints[:, :, :2]

        # percentage of visible keypoints (out of total # of keypoints per part)
        visible = (np.count_nonzero(visibility == 2, axis=1) / 21) * 100
        
        vis['hand'].append(visible)
        vis['thumb'].append((np.count_nonzero(visibility[:, 1:5] == 2, axis=1) / 4) * 100)
        vis['pointer'].append((np.count_nonzero(visibility[:, 5:9] == 2, axis=1) / 4) * 100)
        vis['middle'].append((np.count_nonzero(visibility[:, 9:13] == 2, axis=1) / 4) * 100)
        vis['ring'].append((np.count_nonzero(visibility[:, 13:17] == 2, axis=1) / 4) * 100)
        vis['pinky'].append((np.count_nonzero(visibility[:, 17:21] == 2, axis=1) / 4) * 100)

        # percentage of occluded keypoints (out of total number of labeled keypoints per part)
        occluded = zero_divide(np.count_nonzero(visibility == 1, axis=1), np.count_nonzero(visibility >= 1, axis=1)) * 100 
        
        occ['hand'].append(occluded)
        occ['thumb'].append(zero_divide(np.count_nonzero(visibility[:, 1:5] == 1, axis=1), np.count_nonzero(visibility[:, 1:5] >= 1, axis=1)) * 100)
        occ['pointer'].append(zero_divide(np.count_nonzero(visibility[:, 5:9] == 1, axis=1), np.count_nonzero(visibility[:, 5:9] >= 1, axis=1)) * 100)
        occ['middle'].append(zero_divide(np.count_nonzero(visibility[:, 9:13] == 1, axis=1), np.count_nonzero(visibility[:, 9:13] >= 1, axis=1)) * 100)
        occ['ring'].append(zero_divide(np.count_nonzero(visibility[:, 13:17] == 1, axis=1), np.count_nonzero(visibility[:, 13:17] >= 1, axis=1)) * 100)
        occ['pinky'].append(zero_divide(np.count_nonzero(visibility[:, 17:21] == 1, axis=1), np.count_nonzero(visibility[:, 17:21] >= 1, axis=1)) * 100)

        # occluded = (np.count_nonzero(visibility == 1, axis=1) / 21) * 100
        
        # occ['hand'].append(occluded)
        # occ['thumb'].append((np.count_nonzero(visibility[:, 1:5] == 1, axis=1) / 4) * 100)
        # occ['pointer'].append((np.count_nonzero(visibility[:, 5:9] == 1, axis=1) / 4) * 100)
        # occ['middle'].append((np.count_nonzero(visibility[:, 9:13] == 1, axis=1) / 4) * 100)
        # occ['ring'].append((np.count_nonzero(visibility[:, 13:17] == 1, axis=1) / 4) * 100)
        # occ['pinky'].append((np.count_nonzero(visibility[:, 17:21] == 1, axis=1) / 4) * 100)


        # percentage of labeled keypoints (out of total # of keypoints per part)
        labeled = (np.count_nonzero(visibility >= 1, axis=1) / 21) * 100
        lab['hand'].append(labeled)
        lab['thumb'].append((np.count_nonzero(visibility[:, 1:5] >= 1, axis=1) / 4) * 100)
        lab['pointer'].append((np.count_nonzero(visibility[:, 5:9] >= 1, axis=1) / 4) * 100)
        lab['middle'].append((np.count_nonzero(visibility[:, 9:13] >= 1, axis=1) / 4) * 100)
        lab['ring'].append((np.count_nonzero(visibility[:, 13:17] >= 1, axis=1) / 4) * 100)
        lab['pinky'].append((np.count_nonzero(visibility[:, 17:21] >= 1, axis=1) / 4) * 100)


        if args.use_gt: 
            distances = np.sqrt(np.sum((keypoints - pose_preds) ** 2, axis=2))
            bbox = np.asarray([max(a["bbox"][2], a["bbox"][3]) for a in anns]).reshape((-1, 1))

        else:
            bbox = []
            distances = []
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

                distances = np.sqrt(np.sum((keypoints - pose_preds) ** 2, axis=2))

                bbox.extend([max(a[2], a[3]) for a in gt_bboxes])

            # unmatched keypoints dist is inf
            if len(idxs_pred) < len(anns):
                filler = np.zeros((len(anns) - len(idxs_pred), 21))
                filler.fill(np.inf)

                if len(idxs_pred) >= 1:
                    distances = np.vstack([distances, filler])
                else:
                    distances = filler

                bbox.extend([0 for i in range(len(anns) - len(idxs_pred))])


            bbox = np.asarray(bbox).reshape((-1, 1))


        # per hand PCK calculation
        pcks['hand'].append(np.mean(distances <= args.alpha * bbox, axis=1))
        pcks['thumb'].append(np.nanmean(distances[:, 1:5] <= args.alpha * bbox, axis=1))
        pcks['pointer'].append(np.nanmean(distances[:, 5:9] <= args.alpha * bbox, axis=1))
        pcks['middle'].append(np.nanmean(distances[:, 9:13] <= args.alpha * bbox, axis=1))
        pcks['ring'].append(np.nanmean(distances[:, 13:17] <= args.alpha * bbox, axis=1))
        pcks['pinky'].append(np.nanmean(distances[:, 17:21] <= args.alpha * bbox, axis=1))


    for lst in [vis, lab, occ, pcks]:
        for key, value in lst.items():
            lst[key] = np.hstack(value)

    for area in ['hand', 'thumb', 'pointer', 'middle', 'ring', 'pinky']:

        # don't use visible bc it's very much like labled, since the majority of labled keypoints are visible
        v = [x for x in zip(vis[area].tolist(), pcks[area].tolist())]
        v = sorted(v, key=lambda x: x[0])

        o = [x for x in zip(occ[area].tolist(), pcks[area].tolist())]
        o = sorted(o, key=lambda x: x[0])

        l = [x for x in zip(lab[area].tolist(), pcks[area].tolist())]
        l = sorted(l, key=lambda x: x[0])



        ## MATPLOTLIB GRAPHING 
        # plt.plot([ll[0] for ll in l], [p[1] for p in l], 'b+', [oo[0] for oo in o], [p[1] for p in o], 'gx')
        # # plt.plot([vv[0] for vv in v], [p[1] for p in v], 'ro')

        # plt.xlabel("% visible, % labled")
        # plt.ylabel("PCK@" + str(args.alpha))

        # if args.use_gt:
        #     plt.savefig('gt_bbox_vis_labeled_' + area + '.png')
        # else:
        #     plt.savefig('dt_bbox_vis_labeled_' + area + '.png')
        # plt.clf()



        fig = go.Figure()

        fig.add_trace(go.Scatter(
                x=[ll[0] for ll in l], y=[p[1] for p in l],
                name='Percent Labeled Keypoints',
                mode='markers',
                marker_color='rgba(152, 0, 0, .8)',
                marker_symbol='circle-open'
            ))

        fig.add_trace(go.Scatter(
            x=[oo[0] for oo in o], y=[p[1] for p in o],
            name='Percent Occluded Keypoints of Labled',
            marker_color='rgba(60, 200, 255, .8)',
            marker_symbol='circle-open'
        ))

        # Set options common to all traces with fig.update_traces
        fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
        fig.update_layout(title='',
                          yaxis_zeroline=False, xaxis_zeroline=False)
        fig.update_xaxes(title_text='Percentage')
        fig.update_yaxes(title_text='PCK@' + str(args.alpha))

        if args.use_gt:
            fig.write_html('html/gt_bbox_vis_labeled_' + area + ("_test" if args.test else "") + '.html')
        else:
            fig.write_html('html/dt_bbox_vis_labeled_' + area + ("_test" if args.test else "") + '.html')
    


if __name__ == '__main__':
    main()