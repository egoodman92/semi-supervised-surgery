from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from pyprojroot import here
proj_path = here()
sys.path.append(str(proj_path / "MULTITASK_FILES/KEYPOINTS_FILES/surgery-hand-detection-new"))
sys.path.append(str(proj_path / "MULTITASK_FILES/KEYPOINTS_FILES/surgery-hand-detection-new/deep-high-resolution-net.pytorch/lib"))

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

print("\nWe are using torch version", torch.__version__)
print("We are using torchvision version", torchvision.__version__, "\n")

import sys
sys.path.append("../MULTITASK_FILES/KEYPOINTS_FILES/surgery-hand-detection-new/deep-high-resolution-net.pytorch/lib")
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











DEFAULT_CATEGORIES = ['cutting', 'tying', 'suturing', 'background']
DETECTION_CLASSES = ['bovie', 'forceps', 'needledriver', 'hand']

import sys
sys.path.insert(0, str(proj_path / "MULTITASK_FILES/TSM_FILES/"))
sys.path.append(str(proj_path / 'MULTITASK_FILES/RETINANET_FILES/src/pytorch-retinanet/'))

from dataset import * #imports dataloaders from TSM
SurgeryDataset.categories = DEFAULT_CATEGORIES
from train import get_train_val_data_loaders, run_epoch
from model import get_model_name, save_model, save_results, get_model
from barbar import Bar

import torch.nn as nn
from torch.utils.data import DataLoader

import cv2
import utils
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import json
import csv
from collections import defaultdict
import shutil
import timeit
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Surgery Hand and Keypoint Detection on Video')
    parser.add_argument('--vid_name', type=str, default="hyw7Ue6oW8w.mp4")
    parser.add_argument('--directory', type=str, default="../produced_videos/")
    parser.add_argument('--keypoints', action='store_true')
    parser.add_argument('--whole_dir', action='store_true')
    parser.add_argument('--cfg', type=str, default=str(proj_path / "MULTITASK_FILES/KEYPOINTS_FILES/surgery-hand-detection-new/keypoints.yaml"))

    parser.add_argument('--multitaskmodel_loc', type=str, default=proj_path / "logs/20210514_multitaskmodel_R2p1d/20210514_multitaskmodel_R2p1d_130_incomplete.pt")


    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--epsilon', type=int, default=100)
    parser.add_argument('--num_smoothing_frames', type=int, default=15)
    parser.add_argument('--percentage_hits', type=float, default=.4)

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


def get_test_data_loaders(segments_df, batch_size, data_dir='data/', model='TSM', pre_crop_size=352, segment_length=5,
                                                                    aug_method='val'):
    df = segments_df.sort_values(by=['video_id', 'start_seconds'])
    test_dataset = SurgeryDataset(df, data_dir=data_dir, mode='test', model=model, balance=False,
                                   pre_crop_size=pre_crop_size, aug_method=aug_method)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)
    return test_data_loader



def get_video_path(video_id, data_dir='data/'):
    return os.path.join(data_dir + video_id + ".mp4")



def get_video_duration(filename):
    video = cv2.VideoCapture(filename)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    return frame_count, fps



def anns_from_video_ids(video_ids, data_dir, segment_length, offset=0):
    rows = []
    for video_id in video_ids:
        video_path = get_video_path(video_id, data_dir)
        print("Studying video at path", video_path, "\n")
        if not os.path.exists(video_path):
            print("Video not downloaded: %s" % video_id)
            continue
        frame_count, fps = get_video_duration(video_path)
        num_anns = int(frame_count / fps / segment_length)
        for i in range(num_anns):
            start_seconds = offset + i * segment_length
            label = 'background'
            row = {'start_seconds': start_seconds,
                   'video_id': video_id,
                   'end_seconds': start_seconds + segment_length,
                   'duration': segment_length,
                   'label': label,
                   'category': label}
            rows.append(row)
    anns_df = pd.DataFrame(rows)
    return anns_df










from retinanet import model_3_heads_new
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

def remove_from_cache_by_scores(cache, cache_scores, cache_classes):

    #filters the buffer cache of comparison detections
    #based on the cache_scores

    new_cache = []
    new_scores = []
    new_classes = []

    for f in range(len(cache)): #iterate through frames

        frame_cache = torch.empty(0,4).cuda() #[]
        frame_scores = torch.empty(0).cuda() #[]
        frame_classes = torch.empty(0).cuda() #[]

        for d in range(cache[f].shape[0]): #iterate through dets in a frame

            if cache_scores[f][d] > 0.5:
                frame_cache = torch.cat((frame_cache, cache[f][d].unsqueeze(0)), dim=0)
                frame_scores = torch.cat((frame_scores, cache_scores[f][d].unsqueeze(0)), dim=0)
                frame_classes = torch.cat((frame_classes, cache_classes[f][d].unsqueeze(0)), dim=0)
                #frame_cache.append(cache[f][d])
                #frame_scores.append(cache_scores[f][d])
                #frame_classes.append(cache_classes[f][d])

        if len(frame_cache) > 0:
            #frame_cache = torch.stack(frame_cache)
            new_cache.append(frame_cache)
            #frame_scores = torch.stack(frame_scores)
            new_scores.append(frame_scores)
            #frame_classes = torch.stack(frame_classes)
            new_classes.append(frame_classes)
        else:
            new_cache.append(torch.empty(0,4))
            new_scores.append(torch.empty(0))
            new_classes.append(torch.empty(0))

    return new_cache, new_scores, new_classes



def compare_detection_to_cache(detection, cache_classes, cache, epsilon=100):

    #this function compares a single detection to all the detections in the surrounding cache
    #returns num_proximate, the number of detections in the cache we're close to.

    num_proximate = 0

    class_votes = []

    for frame_dets, frame_classes in zip(cache, cache_classes):
        for det, det_class in zip(frame_dets, frame_classes):
            difference = torch.sum(abs(detection - det))
            if difference < epsilon:
                num_proximate += 1
                class_votes.append(int(det_class.cpu()))
                break

    return num_proximate, class_votes



def smooth_detections(batch_nms_scores, batch_nms_class, batch_transformed_anchors, num_smoothing_frames=5, percentage_hits = 0.7, epsilon=100):

    #take nms_scores, nms_classes, and transformed anchors
    #this is done on a batch level of 64
    #if you want to suppress a detection, make nms_score = 10**6
    #you keep a detection if you can find a certain number of detections
    #within the the buffer cache

    num_deleted_detections = 0
    num_identity_changes = 0

    batch_transformed_anchors, batch_nms_scores, batch_nms_class = remove_from_cache_by_scores(batch_transformed_anchors, batch_nms_scores, batch_nms_class)

    for frame_no in range(len(batch_nms_scores)):

        #print("\n\nStudying frame", frame_no)
        cur_anchors = batch_transformed_anchors[frame_no]
        cur_class = batch_nms_class[frame_no]
        cur_scores = batch_nms_scores[frame_no]

        frame_anchors_to_keep = []
        for d in range(cur_anchors.shape[0]):


            #print("Studying anchor {} {}, class {}, cur_scores {}".format(d, cur_anchors[d], cur_class[d], cur_scores[d]))
            #print("Cache is between", max(0, frame_no-int(num_smoothing_frames/2)), min(len(batch_transformed_anchors), frame_no+int(num_smoothing_frames/2)+1))

            #take a subset of your transformed anchors to compare against for filter, and use remove_from_cache_by_scores to make sure you only have confident in your cache
            cache = batch_transformed_anchors[max(0, frame_no-int(num_smoothing_frames/2)) : min(len(batch_transformed_anchors), frame_no+int(num_smoothing_frames/2))]
            cache_scores = batch_nms_scores[max(0, frame_no-int(num_smoothing_frames/2)) : min(len(batch_transformed_anchors), frame_no+int(num_smoothing_frames/2))]
            cache_classes = batch_nms_class[max(0, frame_no-int(num_smoothing_frames/2)) : min(len(batch_transformed_anchors), frame_no+int(num_smoothing_frames/2))]

            #cache = remove_from_cache_by_scores(cache, cache_scores, cache_classes)

            num_proximate, class_votes = compare_detection_to_cache(cur_anchors[d], cache_classes, cache, epsilon=100)
            #print("Comparing class votes {} to actual class {}".format(class_votes, cur_class[d]))
            if cur_class[d] != int(max(set(class_votes), key=class_votes.count)) and int(cur_class[d].cpu()) != 3 and int(max(set(class_votes), key=class_votes.count)) != 3:
                #print(int(cur_class[d].cpu()), int(max(set(class_votes), key=class_votes.count)))
                cur_class[d] = int(max(set(class_votes), key=class_votes.count))
                #print("Identity change!")
                num_identity_changes += 1

            #set the score equal to a very high nonreal number so you don't plot it
            #print("NUM PROXIMATE", num_proximate)

            #print("LENGTH CACHE {} AND NUM SMOOTHING FRAMES {}".format(len(cache), num_smoothing_frames))
            if num_proximate < percentage_hits * len(cache):
                batch_nms_scores[frame_no][d] = 10**6
                num_deleted_detections += 1


    return batch_nms_scores, batch_nms_class, batch_transformed_anchors


#color scheme: hands(gold), bovie(red), needledriver(blue), forceps(green)
annot_to_color = {0 : (255, 0, 0), 1 : (0, 255, 0), 2 : (0, 0, 255), 3 : (255, 215, 0) }


def inference(test_data_loader, vid_name, multitaskmodel_loc, directory, args):

    #transformation for poses
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    #update the configs of the keypoints inference
    update_config(cfg, args)

    #initialize keypoints model, load pretrained weights, and send to GPU and eval
    pose_model = eval(cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)
    if cfg.TEST.MODEL_FILE:
        print('=> loading keypoints model from {}'.format("MULTITASK_FILES/KEYPOINTS_FILES/surgery-hand-detection-new/" + cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(str(proj_path) + "/MULTITASK_FILES/KEYPOINTS_FILES/surgery-hand-detection-new/" + cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')
    pose_model.to(CTX)
    pose_model.eval()


    #used for measuring the inference time
    retinanet = model_3_heads_new.resnet50(num_classes=4)
    retinanet = torch.load(multitaskmodel_loc) 
    print('=> loading multitaskmodel model from {}'.format(multitaskmodel_loc))
    retinanet.to(CTX)

    tool_model = retinanet
    if torch.cuda.is_available():
        tool_model = tool_model.cuda()
        tool_model = torch.nn.DataParallel(tool_model).cuda()
        tool_model.eval()

    #this is just to get the dimensions of the dataset we're working with! only done at very beginning
    for iter_num, (data_action, record_ids, action_labels) in enumerate(test_data_loader):
        #print("Grabbing dimensions")
        original_dimensions = record_ids[1]
        #print("Original dimensions are", record_ids[1])
        break

    data_action = (data_action.view((-1, 3) + data_action.size()[-2:]))
    b, c, height, width = data_action.shape
    IMAGE_SIZE = [height, width] #this is the way its written for hand keypoints, so we write it like this too
    
    #video for superposition of detections/actions
    video = directory + vid_name
    video = cv2.VideoCapture(video)
    fps_video = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    #params for output video and json
    out_video = directory + vid_name[:-4] + "_detections.mp4"
    if args.smooth:
        out_video = directory + vid_name[:-4] + "_detections" + "_smooth_nsf{}e{}ph{}.mp4".format(args.num_smoothing_frames, args.epsilon, args.percentage_hits)

    fps = 13 #frame rate defined by model
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_tracked = cv2.VideoWriter(out_video, fourcc, fps, (int(width), int(height)))
    output_json = defaultdict(list)

    #params for clean out video for tracking
    clean_out_video = directory + vid_name[:-4] + "_nodetections.mp4"
    fps = 13
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_clean = cv2.VideoWriter(clean_out_video, fourcc, fps, (int(width), int(height)))


    print('''

    Video has following parameters
    Frame Height : {}
    Frame Width : {}
    FPS : {}
    Number of Frames : {}
    Duration (seconds) : {}

    '''.format(height, width, fps_video, frame_count, frame_count/fps_video))

    start_time = timeit.default_timer()
    print("Starting inference on video {} at time {}\n".format(vid_name, start_time))

    for iter_num, (data_action, record_ids, action_labels) in enumerate(test_data_loader):

        with torch.no_grad():

            if iter_num == 3:
                break

            #reshape data and forward pass!
            data_action = (data_action.view((-1, 3) + data_action.size()[-2:]))
            print("Studying batch", iter_num, data_action.shape)

            #forward through model
            batch_nms_scores, batch_nms_class, batch_transformed_anchors, action_logits = tool_model(data_action)

            #smooths outputs if you want
            if args.smooth:
                batch_nms_scores, batch_nms_class, batch_transformed_anchors =\
                    smooth_detections(batch_nms_scores, batch_nms_class, batch_transformed_anchors, args.num_smoothing_frames, args.percentage_hits, args.epsilon)


            #go frame by frame through output and add to video
            for frame_no in range(len(batch_nms_scores)):

                frame_detections = []

                #get scores, anchors, and classes for this particular frame
                nms_scores = batch_nms_scores[frame_no]
                transformed_anchors = batch_transformed_anchors[frame_no]
                nms_class = batch_nms_class[frame_no]

                #filter for scores >0.5. The last clause is used for filtering as a proxy value
                idxs = np.where(np.logical_and(nms_scores.cpu() >= .5, nms_scores.cpu() < 10**6))

                #some stuff so that you can plot on top of the frame (I think mostly because opencv is annoying)
                frame = data_action.squeeze().cuda().float()[frame_no, :, :, :].cpu().numpy()
                frame = np.transpose(frame, (1,2,0))
                frame = np.array(255*(frame.copy() *np.array([[[0.2650, 0.2877, 0.3311]]]) + np.array([[[0.3051, 0.3570, 0.4115]]])), dtype = np.uint8  )
                image_pose = frame.copy()


                video_clean.write(np.uint8(frame))

                centers, scales = [], []
                if len(idxs[0]) > 0:
                    for idx in idxs[0]:
                        a = float(transformed_anchors[idx].detach()[0])
                        b = float(transformed_anchors[idx].detach()[1])
                        c = float(transformed_anchors[idx].detach()[2])
                        d = float(transformed_anchors[idx].detach()[3])
                        e = float(nms_class[idx].detach()) #this last coordinate is the ID

                        #draws the actual rectangle
                        cv2.rectangle(frame, (int(a), int(b)), (int(c), int(d)), color=annot_to_color[int(e)], thickness=3)

                        #for doing pose predictions, need center and scale of each box
                        if int(e) == 3 and args.keypoints: #filter by hand label

                            center, scale = box_to_center_scale([a, b, c, d], IMAGE_SIZE[0], IMAGE_SIZE[0])

                            pose_preds = get_pose_estimation_prediction(pose_model, image_pose, [center], [scale], transform=pose_transform)

                            for coords in pose_preds:

                                for i, coord in enumerate(coords):
                                    x_coord, y_coord = float(max(0, coord[0])), float(max(0, coord[1]))
                                    if not (x_coord == 0 and y_coord == 0):
                                        x_coord, y_coord = int(x_coord), int(y_coord)
                                        cv2.circle(frame, (x_coord, y_coord), 4, keypoint_colors[str(i + 1)], -1)
                                        cv2.putText(frame, str(i + 1), (x_coord - 4, y_coord - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                            pose_preds = pose_preds.tolist()
                        else:
                            pose_preds = None


                        object_dict = {DETECTION_CLASSES[int(e)] : {"bbox": [a, b, c, d, float(nms_scores[idx])], "keypoints": pose_preds}}
                        frame_detections.append(object_dict)



                #used for labeling action
                cur_action = DEFAULT_CATEGORIES[int(torch.argmax(action_logits))]
                cur_action_prob = torch.max(action_logits)
                cv2.putText(frame, cur_action + " " + str(float(cur_action_prob.data))[:5], (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255) )

                #add all these annotations to the video or json
                output_json[int(64*iter_num + frame_no)] = {"actions" : action_logits.detach().cpu().tolist()[0], "detections" : frame_detections}

                video_tracked.write(np.uint8(frame))



    #calculate how fast our inference was!
    elapsed = timeit.default_timer() - start_time
    print('\n ...inference complete after {} !'.format(elapsed))
    
    #release the video and the json!
    video_tracked.release()
    video_clean.release()

    with open(directory + vid_name[:-4] + "_detections.json", "w") as outfile:
        json.dump(output_json, outfile)
    print("\nOutput video is ", out_video)
    
    return frame_count/fps_video, elapsed, original_dimensions

def inference_txt(directory, vid_name, inference_outputs, keypoints):

    print("FINISHED INFERENCE!")
    print("\nVideo was {} seconds, and inference was performed in {} seconds\n".format(inference_outputs[0], inference_outputs[1]))
    output = [vid_name, inference_outputs[0], inference_outputs[1], keypoints]

    with open(directory + vid_name + '_keypoints_' + str(keypoints) + '_times.txt', 'w') as filehandle:
        for listitem in output:
            print("WRITING!!!", directory + vid_name + '_keypoints_' + str(keypoints) + '_times.txt')
            filehandle.write('%s ' % listitem)


def main():
    args = parse_args()

    print("\nStore keypoints is", args.keypoints, "\n")

    if args.whole_dir:
        print("Doing whole directory!")
        for f in os.listdir(args.directory):
            print("NOW STUDYING FILE", f)
            if f[:-4] + "_detections.json" not in os.listdir(args.directory) and f.endswith(".mp4") \
            and not f.endswith("detections.mp4") and not f.endswith(".json"):
                print("Not studied yet!")
                segments_df = anns_from_video_ids([f[:-4]], args.directory, segment_length=5)
                test_data_loader = get_test_data_loaders(segments_df, batch_size=1, data_dir = args.directory)
                inference_outputs = inference(test_data_loader, f, args.multitaskmodel_loc, args.directory, args)
                inference_txt(args.directory, f[:-4], inference_outputs, args.keypoints)

    else:
        segments_df = anns_from_video_ids([args.vid_name[:-4]], args.directory, segment_length=5)
        test_data_loader = get_test_data_loaders(segments_df, batch_size=1, data_dir = args.directory)
        inference_outputs = inference(test_data_loader, args.vid_name, args.multitaskmodel_loc, args.directory, args)

        inference_txt(args.directory, args.vid_name[:-4], inference_outputs, args.keypoints)

    #print("Returning inference outputs", inference_outputs)

    #print("\nVideo was {} seconds, and inference was performed in {} seconds\n".format(inference_outputs[0], inference_outputs[1]))
    #output = [args.vid_name[:-4], inference_outputs[0], inference_outputs[1], \
    #         float(inference_outputs[2][0]), float(inference_outputs[2][1]), float(inference_outputs[2][2])]

    #with open(args.vid_name[:-4]+'_times.txt', 'w') as filehandle:
    #    for listitem in output:
    #        filehandle.write('%s ' % listitem)

if __name__ == '__main__':
    main()

