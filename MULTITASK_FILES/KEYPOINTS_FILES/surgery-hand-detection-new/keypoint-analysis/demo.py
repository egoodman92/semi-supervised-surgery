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

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import os
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

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
    '__background__', 'right hand', 'left hand'
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
        if (pred_score > threshold) and (pred_class == 'right hand' or pred_class == 'left hand'):
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

    # bottom_left_corner = box[0]
    # top_right_corner = box[1]
    # box_width = top_right_corner[0]-bottom_left_corner[0]
    # box_height = top_right_corner[1]-bottom_left_corner[1]
    # bottom_left_x = bottom_left_corner[0]
    # bottom_left_y = bottom_left_corner[1]
    # center[0] = bottom_left_x + box_width * 0.5
    # center[1] = bottom_left_y + box_height * 0.5

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


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--outputDir', type=str, default='../inference_train/')
    parser.add_argument('--writeBoxFrames', action='store_true')

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
    pose_dir = prepare_output_dirs(args.outputDir)
    csv_output_rows = []

    bbox_cfg = get_cfg()
    bbox_cfg.merge_from_file("../bbox.yaml")
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
    

    # coco = json.load(open("../data/coco/annotations/hand_keypoints_val.json", "r"))
    # max_img_id = max([x["id"] for x in coco["images"]])
    # max_ann_id = max([x["id"] for x in coco["annotations"]])
    # for base in ["../data/coco/images/val"]:
    for entry in os.scandir("../data/coco/images/val"):

    # gt = COCO("../data/coco/annotations/hand_keypoints_val.json")

    # video = cv2.VideoCapture("wZTMcbt85J4.mp4")
    # width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # fps = video.get(cv2.CAP_PROP_FPS)
    # print(fps)
    # data = {}

    # fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    # video_tracked = cv2.VideoWriter('tracked_val_final.mp4', fourcc, fps, (int(width), int(height)))

    # frame_num = 0
    # while video.isOpened():    
    #     print("frame num ", frame_num)
    #     _, frame = video.read()
    #     if frame is None or frame.size == 0:
    #         break

    # for img_id in gt.getImageIds():

        # path = "../data/coco/images/" + gt.loadImgs([img_id])[0]['file_name']

        img = cv2.imread(entry.path)
        # img = frame
        image_debug = img.copy()
        image_pose = img.copy()
        pred_boxes = box_model(img)['instances'].pred_boxes

        centers = []
        scales = []
        for box in pred_boxes:
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            centers.append(center)
            scales.append(scale)

        if len(pred_boxes) == 0:
            # frame_num += 1
            # video_tracked.write(image_debug)
            continue

        now = time.time()
        pose_preds = get_pose_estimation_prediction(pose_model, image_pose, centers, scales, transform=pose_transform)
        then = time.time()
        # print("Find person pose in: {} sec".format(then - now))

        # sz = Image.open(entry.path)
        # width, height = sz.size
        # coco['images'].append({
        #     "file_name": os.path.split(entry.path)[1],
        #     "height": height,
        #     "width": width,
        #     "id": max_img_id + 1
        #     })

        # for i, box in enumerate(pred_boxes):
        #     converted = {
        #         'image_id': max_img_id + 1,
        #         'segmentation': [],
        #         'iscrowd': 0,
        #         'id': max_ann_id + 1,
        #         'category_id': 1
        #         }

        #     num_keypoints = 0
        #     converted['keypoints'] = [0 for _ in range(3*21)]
        #     cur_keypoints = pose_preds[i]

        #     for j, (x, y) in enumerate(cur_keypoints):
        #         num_keypoints += 1
        #         idx = j * 3
        #         converted['keypoints'][idx] = float(x)
        #         converted['keypoints'][idx + 1] = float(y)
        #         converted['keypoints'][idx + 2] = 2

        #     converted['num_keypoints'] = num_keypoints

# ERROR
# NEEDS TO BE X, Y, W, H
#             converted['bbox'] = [float(box[0]),
#                                     float(box[1]),
#                                     float(box[2]),
#                                     float(box[3])]

#             converted['area'] = float((converted['bbox'][2] - converted['bbox'][0]) * (converted['bbox'][3] - converted['bbox'][1]))
# # END ERROR
        #     converted["bbox"] = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])] # x y w h
        #     converted["area"] = float((box[2] - box[0]) * (box[3] - box[1]))
        #     coco['annotations'].append(converted)
        #     max_ann_id += 1

        # max_img_id += 1
        
        preds = []
        for coords in pose_preds:
            # Draw each point on image
            preds.append({"keypoints":[]})
            for i, coord in enumerate(coords):
                x_coord, y_coord = float(max(0, coord[0])), float(max(0, coord[1]))
                preds[-1]["keypoints"].append((x_coord if x_coord > 0 else 0, y_coord if y_coord > 0 else 0))


                if not (x_coord == 0 and y_coord == 0):
                    x_coord, y_coord = int(x_coord), int(y_coord)
                    cv2.circle(image_debug, (x_coord, y_coord), 4, keypoint_colors[str(i + 1)], -1)
                    cv2.putText(image_debug, str(i + 1), (x_coord - 4, y_coord - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


        for it, box in enumerate(pred_boxes):
            preds[it]["bbox"] = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            cv2.rectangle(image_debug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 255, 0),
                              thickness=3) 

        cv2.imwrite("../val_evaluation_prev/" + os.path.split(entry.path)[1], image_debug)

    #     video_tracked.write(image_debug)
    #     data[frame_num] = preds
    #     frame_num += 1


        
    # video.release()
    # video_tracked.release()
        
        # print("saved image")

    # json.dump(data, open("wZTMcbt85J4.json", "w"))

    # json.dump(coco, open("../hand_keypoints_val_bootstrap.json", "w"))

if __name__ == '__main__':
    main()
