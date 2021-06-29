# modified from original source by Krishna Patel
# source: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/demo/inference.py

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

print("We are using torch version", torch.__version__)
print("We are using torchvision version", torchvision.__version__)

import sys
sys.path.append("./deep-high-resolution-net.pytorch/lib")
import time

from models import pose_hrnet
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import os

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


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
    '__background__', 'hand',
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
    parser = argparse.ArgumentParser(description='Surgery Hand and Keypoint Detection on Video')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--bb_cfg', type=str, required=True)
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--produce_vid', action='store_true')
    parser.add_argument('--out_json', type=str, default='out.json')
    parser.add_argument('--tracking', action='store_true')
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

    bbox_cfg = get_cfg()
    bbox_cfg.merge_from_file(args.bb_cfg)
    box_model = DefaultPredictor(bbox_cfg)

    pose_model = eval(cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    video = cv2.VideoCapture(args.video)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)
    data = {}

    if args.produce_vid:
        fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
        video_tracked = cv2.VideoWriter('predictions.mp4', fourcc, fps, (int(width), int(height)))

    frame_num = 0
    while video.isOpened():    
        print("Performing Inference on Frame Number ", frame_num, end='\r')
        _, frame = video.read()
        if frame is None or frame.size == 0:
            break

        img = frame
        if args.produce_vid:
            image_debug = img.copy()
        image_pose = img.copy()
        predictions = box_model(img)['instances']
        pred_boxes = predictions.pred_boxes

        centers = []
        scales = []
        for box in pred_boxes:
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            centers.append(center)
            scales.append(scale)

        if len(pred_boxes) == 0:
            frame_num += 1
            if args.produce_vid:
                video_tracked.write(image_debug)
            continue

        now = time.time()
        pose_preds = get_pose_estimation_prediction(pose_model, image_pose, centers, scales, transform=pose_transform)
        then = time.time()
        
        preds = []

        for coords in pose_preds:

            preds.append({"keypoints":[]})
            for i, coord in enumerate(coords):
                x_coord, y_coord = float(max(0, coord[0])), float(max(0, coord[1]))
                preds[-1]["keypoints"].append((x_coord if x_coord > 0 else 0, y_coord if y_coord > 0 else 0))

                if not (x_coord == 0 and y_coord == 0):
                    x_coord, y_coord = int(x_coord), int(y_coord)
                    if args.produce_vid:
                        cv2.circle(image_debug, (x_coord, y_coord), 4, keypoint_colors[str(i + 1)], -1)
                        cv2.putText(image_debug, str(i + 1), (x_coord - 4, y_coord - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


        for it, box in enumerate(pred_boxes):
            preds[it]["bbox"] = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            if args.tracking:
                preds[it]["bbox"].append(float(predictions.scores[it].detach().cpu().numpy()))
            if args.produce_vid:
                cv2.rectangle(image_debug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 255, 0),
                              thickness=3) 

        if args.produce_vid:
            video_tracked.write(image_debug)
        data[frame_num] = preds
        frame_num += 1
        
    video.release()
    if args.produce_vid:
        video_tracked.release()
        

    json.dump(data, open(args.out_json, "w"))


if __name__ == '__main__':
    main()
