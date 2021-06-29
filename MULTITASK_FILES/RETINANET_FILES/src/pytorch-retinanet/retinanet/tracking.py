import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import sys
import cv2
import json
import os
import csv
import math

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from IPython.display import HTML
import shutil
import datetime

from skimage.io import imread
from skimage import filters, img_as_float
from skimage import img_as_ubyte

from PIL import Image

from pytube import YouTube

from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer, InferenceDataset

from retinanet import csv_eval
from pytube import YouTube

def load_frames(imgs_dir):
    frames = [img_as_float(imread(os.path.join(imgs_dir, frame), plugin='matplotlib')) \
              for frame in sorted(os.listdir(imgs_dir))]
    return frames

def draw_caption(image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] + 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

def correct_color(img):
    out = np.ndarray(img.shape)
    out[:, :, 0] = img[:, :, 2]
    out[:, :, 1] = img[:, :, 1]
    out[:, :, 2] = img[:, :, 0]
    return out

def load_bboxes(gt_path):
    bboxes = []
    with open(gt_path) as f:
        for line in f:
            x, y, w, h = line.split(',')
            bboxes.append((int(x), int(y), int(w), int(h)))
    return bboxes

def animated_frames(frames, figsize=(10,8), interval_ms=100):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(frames[0])

    def animate(i):
        im.set_array(frames[i])
        return [im,]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=interval_ms, blit=True)

    return ani

class FrameExtractor():
    '''
    Class used for extracting frames from a video file.
    '''
    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap = cv2.VideoCapture(video_path)
        self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
        self.duration = self.n_frames / self.fps
        
    def get_n_images(self, every_x_frame):
        return math.floor(self.n_frames / every_x_frame) + 1
    
    # Sequences: a list of tupes (n_prior, center, n_post), can be formatted as seconds or frames (depending on what mode is set to)
    def extract_frames(self, every_x_frame, sequences=None, mode=None, orig_frames=None):
        if sequences is None:
            return []
        
        segments = [[] for i in range(len(sequences))]
        requested_frames = []
        for f_before, f_center, f_after in sequences:
            if mode == 's':
                central_frame = int(f_center * self.fps)
                requested_frames.append((central_frame - int(f_before * self.fps), int(central_frame + f_after * self.fps)))
            else:
                if orig_frames is not None:
                    central_frame = int(f_center * self.n_frames / orig_frames)
                else:
                    central_frame = f_center
                requested_frames.append((central_frame - f_before, central_frame + f_after))

        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)
        
        frame_cnt = 0
        frame_begin = min([first_frame for first_frame, last_frame in requested_frames])
        frame_end = max([last_frame for first_frame, last_frame in requested_frames])

        while self.vid_cap.isOpened():
            
            success,image = self.vid_cap.read()

            if not success:
                break

            frame_cnt += 1

            if frame_cnt < frame_begin:
                continue

            if frame_cnt > frame_end:
                break
            
            if frame_cnt % every_x_frame == 0:
                for idx, interval in enumerate(requested_frames):
                    first_frame, last_frame = interval
                    if first_frame <= frame_cnt <= last_frame:
                        segments[idx].append(correct_color(image).astype(np.uint8))

        self.vid_cap.release()

        return segments, self.fps / every_x_frame
    
def prepare_video(url, resolution=360, filename='tmp', vid_tag=18):
    video = YouTube(url)
    video.streams.filter(file_extension='mp4', res=resolution)
    try:
        query=video.streams.get_by_itag(vid_tag)
        query.download(filename='tmp')
    except Exception as e:
        print('Error: vid_tag invalid. Please use a tag from the following:')
        print(video.streams)

def track_video_inference(model, data_loader, labels, threshold=0.5):
    unnormalize = UnNormalizer()

    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    
    frames = []
    
    for idx, data in enumerate(data_loader):
        # try:
        with torch.no_grad():
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = model(data['img'].cuda().float())
            else:
                scores, classification, transformed_anchors = model(data['img'].float())

            idxs = np.where(scores.cpu() > threshold)

            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(correct_color(img).astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                score = scores[idxs[0][j]].item()
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = labels[int(classification[idxs[0][j]])]
                confidence = score * 100

                draw_caption(img, (x1, y1, x2, y2), '{} {:.1f}%'.format(label_name, confidence))
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            frames.append(img)

        # except Exception as e:
        #     print(e)
        #     continue

    return frames

def get_frames_from_url(url, sequences, my_model=None, labels=None, resolution=360, tag=18, every_n_frames=1, mode=None, orig_frames=None):
    print('Downloading video... ', end='')
    prepare_video(url, resolution=resolution, vid_tag=tag)
    print('complete.')
    print('Extracting relevant frames... ', end='')
    extractor = FrameExtractor('tmp.mp4')
    segments, fps = extractor.extract_frames(every_n_frames, sequences, mode=mode, orig_frames=orig_frames) # frames, fps
    os.remove('tmp.mp4')
    print('complete.')
    
    return segments, 1000 / fps

def track_from_url(url, my_model, labels, center, n_before=10, n_after=10, resolution=360, tag=18, every_n_frames=3, threshold=0.5, mode=None, orig_frames=None):
    my_model.training = False
    my_model.eval()
    sequences = [(n_before, center, n_after)]
    segments, interval_ms = get_frames_from_url(url, sequences, resolution=resolution, tag=tag, every_n_frames=every_n_frames, mode=mode, orig_frames=orig_frames)
    
    if len(segments) == 0:
        print('Invalid sequence')
        return
    
    frames = segments[0]
    print('Running inference... ', end='')
    my_dataset = InferenceDataset(frames, transform=transforms.Compose([Normalizer(), Resizer()]))
    my_dataloader = DataLoader(my_dataset, shuffle=False, collate_fn=collater, batch_size=1)
    frames = track_video_inference(my_model, my_dataloader, labels, threshold=threshold)
    print('complete.')
    
    return frames, interval_ms

def draw_bbs(frame, bbs, labels):
    img = frame.copy()
    
    for bb in bbs:
        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color=(0, 0, 255), thickness=2)
        draw_caption(img, (bb[0], bb[1], bb[2], bb[3]), str(labels[bb[4]]))
    
    return img

def get_bbs(my_model, data_loader, labels, show_img=True, threshold=0.5):
    my_model.training = False
    my_model.eval()
    unnormalize = UnNormalizer()
    
    if torch.cuda.is_available():
        my_model = my_model.cuda()
        my_model = torch.nn.DataParallel(my_model).cuda()
    
    result = []
    for idx, data in enumerate(data_loader):
        pad_cols = data['pad_h'][0]
        pad_rows = data['pad_w'][0]
        scale = data['scale'][0]
        with torch.no_grad():
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = my_model(data['img'].cuda().float())
            else:
                scores, classification, transformed_anchors = my_model(data['img'].float())

            idxs = np.where(scores.cpu() > threshold)

            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
            img[img < 0] = 0
            img[img > 255] = 255
            
            img = np.transpose(img, (1, 2, 0))
            rows, cols, _ = img.shape
            rows, cols, = rows - pad_rows, cols - pad_cols
            img = img[:rows, :cols, :]
            img = cv2.cvtColor(correct_color(img).astype(np.uint8), cv2.COLOR_BGR2RGB)
            orig_rows, orig_cols = int(rows // scale), int(cols // scale)
            img = cv2.resize(img, (orig_cols, orig_rows))
            result_img = img.copy()
 
            rows, cols, _ = img.shape

            bbs = np.zeros((0,5))
            for j in range(idxs[0].shape[0]):
                bb = np.zeros((1,5)).astype(np.int)
                bbox = transformed_anchors[idxs[0][j], :]
                score = scores[idxs[0][j]].item()
                bb[0, 0] = int(bbox[0] / scale)
                bb[0, 1] = int(bbox[1] / scale)
                bb[0, 2] = min(int(bbox[2] / scale), cols - 1)                    
                bb[0, 3] = min(int(bbox[3] / scale), rows - 1)
                bb[0, 4] = int(classification[idxs[0][j]])
                bb = bb.astype(np.float32)
                confidence = score * 100
                bbs = np.append(bbs, bb, axis=0)

                if show_img:
                    draw_caption(img, (bb[0, 0], bb[0, 1], bb[0, 2], bb[0, 3]), '{} {:.1f}%'.format(labels[bb[0, 4]], confidence))
                    cv2.rectangle(img, (bb[0, 0], bb[0, 1]), (bb[0, 2], bb[0, 3]), color=(0, 0, 255), thickness=2)

            result.append((result_img, bbs))
            if show_img:
                plt.imshow(img)
                plt.show()
                Image.fromarray(img, 'RGB').save('/home/ubuntu/stephen/curis2020/surgery-tool-detection/logs/my_img.png')
            
    return result


