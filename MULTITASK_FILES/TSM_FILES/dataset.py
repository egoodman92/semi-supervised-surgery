from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle
import torch
import numpy as np
import os
import pandas as pd
import utils
import urllib.request
import cv2
from image import Resize, create_video_clip, augment_raw_frames, test_time_augment
from time import strftime
from time import gmtime
import pickle
import subprocess
import datetime


DEFAULT_CATEGORIES = ['cutting', 'tying', 'suturing', 'background']
SEGMENT_LENGTH = 5



def get_train_val_dfs(anns_path, quality_videos=False, ann_agree=True, k=0):
    df = pd.read_csv(anns_path)

    df = df.sort_values(by=['video_id', 'start_seconds'])

    # TODO: remove and update col names
    #df[df['labeler_2'].isnull()]['labeler_2'] = df[df['labeler_2'].isnull()]['labeler_1']
    df = df[~df['labeler_2'].isnull()]
    # Set label to labeler_3
    # df.loc[~df['labeler_3'].isnull(), 'labeler_2'] = df['labeler_3']
    # only select where labeler_3 matches labeler_2
    df = df[(df['labeler_3'].isnull()) | (df['labeler_2'] == df['labeler_3'])]
    df['label'] = df['labeler_2']
    df['category'] = df['label']
    df['video_name'] = df['video_id']

    videos = None
    # if quality_videos:
    #     videos = utils.videos_by_quality(['good', 'okay'])
    # local miniset
    #videos = ['2j-J2IiHLB8', '3Q5D4vdXOqc', 'Yl3S1UtzmFI']
    # if ann_agree:
    #     df = utils.segments_by_nonexpert_agreement(df)
    t_df, v_df = get_video_split(df, videos, k)
    #print("After train-val split, we have", t_df.shape, v_df.shape)

    return t_df, v_df


def balance_classes(df, categories):
    #print("We are balancing categories", categories)
    min_count = 1e10
    for category in categories:
        count = len(df.loc[df['category'] == category])
        if count < min_count:
            min_count = count
    max_count = min_count

    df2 = pd.DataFrame()
    for category in categories:
        df2 = df2.append(df.loc[df['category'] == category][:max_count])

    return df2

def get_category_based_split(df):
    categories = SurgeryDataset.categories
    df = balance_classes(df, categories)
    num_categories = len(categories)
    split = 0.8
    train_count = int(split * len(df))
    train_per_category_count = int(train_count / num_categories)
    t_df = pd.DataFrame()
    for c in categories:
        tmp_df = df[df['category'] == c].head(train_per_category_count)
        t_df = t_df.append(tmp_df)
    v_df = df[~df['filename'].isin(t_df['filename'])]

    return t_df, v_df


def get_video_split(df, videos=None, k=0):
    if videos is None:
        videos = sorted(list(set(pd.read_csv('data/train.csv')['video_id'])))
    print("In data/train.csv, we have {} videos".format(len(videos)))
    folds = 7
    fold_size = int(len(videos) / folds)
    fold_index = fold_size * k
    #print("fold index: %d" % fold_index)
    videos = videos[fold_index:] + videos[:fold_index]

    #print(videos)

    train_val_split = 1.0 - (1.0 / folds)
    train_count = int(train_val_split * len(videos))
    train_videos = videos[0:train_count]
    val_videos = videos[train_count:]
    print("Train videos: %d, val videos: %d" % (len(train_videos), len(val_videos)))

    t_df = df[df['video_name'].isin(train_videos)]
    v_df = df[df['video_name'].isin(val_videos)]
    if len(t_df) == 0 or len(v_df) == 0:
        train_segs = int(train_val_split * len(df))
        t_df = df.iloc[0:train_segs]
        v_df = df.iloc[train_segs:]

    # t_df = t_df.sample(2000)
    # v_df = v_df.sample(400)


    return t_df, v_df



class SurgeryDataset(Dataset):
    categories = None

    def __init__(self, df, data_dir='data/', mode='train', model='BLV', balance=True,
                 pre_crop_size=256, aug_method='04-20', segment_length=5):
        self.data_dir = data_dir
        self.categories = SurgeryDataset.categories
        if balance:
            self.df = balance_classes(df, self.categories)
        else:
            self.df = df[df['category'].isin(self.categories)]
        self.mode = mode
        self.model = model

        self.pre_crop_size = pre_crop_size
        self.segment_length = segment_length
        self.aug_method = aug_method
        # TODO: Add transformations


    @staticmethod
    def raw_frames_to_input(raw_frames, num_segments=8, method='multi-scale'):
        if method == 'BLV':
            input = np.asarray(raw_frames, dtype=np.uint8)
        elif method == 'multi-scale':
            input = np.asarray(raw_frames[0::8], dtype=np.uint8)
            input = np.expand_dims(input, axis=0)
            for i in range(1, 2):
                new_input = np.asarray(raw_frames[i*2::8], dtype=np.uint8)
                new_input = np.expand_dims(new_input, axis=0)
                input = np.concatenate((input, new_input), axis=0)
            for i in range(2):
                new_input = np.asarray(raw_frames[0::4][i*8:i*8 + 8], dtype=np.uint8)
                new_input = np.expand_dims(new_input, axis=0)
                input = np.concatenate((input, new_input), axis=0)
            for i in range(4):
                new_input = np.asarray(raw_frames[0::2][i*8:i*8 + 8], dtype=np.uint8)
                new_input = np.expand_dims(new_input, axis=0)
                input = np.concatenate((input, new_input), axis=0)
        else:  # uniform
            input = np.asarray(raw_frames[0::8], dtype=np.uint8)
            input = np.expand_dims(input, axis=0)
            for i in range(1, num_segments):
                new_input = np.asarray(raw_frames[i::8], dtype=np.uint8)
                new_input = np.expand_dims(new_input, axis=0)
                #print("Concatenating", input.shape, new_input.shape)
                input = np.concatenate((input, new_input), axis=0)
        return input

    def remote_load_frame(self, video_id, filename):
        img_size = 224
        remote_url = ("https://marvl-surgery.s3.amazonaws.com/frames/%s/%s" % (video_id, filename))
        print("Remote URL: %s" % remote_url)
        with urllib.request.urlopen(remote_url) as req:
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)  # 'Load it as it is'
            if not (img.shape[0] == img_size and img.shape[1] == img_size):
                img = Resize(dsize=(img_size, img_size)).process(img)
            return img

    def get_frames(self, video_id, start_frame, end_frame, num_frames=64):
        # TODO: set as env variable
        #frames_dir = os.path.join(self.data_dir, "frames")
        frames_dir = "/pasteur/data/YoutubeSurgery/images-fps_15"
        if end_frame - start_frame < num_frames:
            print("Warning: not enough frames in segment")
        frames = []
        for i in range(start_frame, start_frame + num_frames):
            filename = (video_id + "-" + "%.9d.jpg") % i
            file_path = os.path.join(frames_dir, video_id, filename)
            if os.path.exists(file_path):
                img = cv2.imread(file_path)[:, :, [2, 1, 0]]
            else:
                img = self.remote_load_frame(video_id, filename)
            frames.append(img)
        return frames

    def get_frames_from_video(self, video_id, video_path, start_seconds, end_seconds, num_frames=64, cache=True):
        #frame_rate = 15
        print("Getting frames the old fashioned way...")
        frame_rate = float(num_frames) / (end_seconds - start_seconds)
        file_path = video_path

        count = 0
        frames = []
        start_frame = frame_rate * start_seconds
        cap = cv2.VideoCapture(file_path)
        playback_fps = cap.get(cv2.CAP_PROP_FPS)
        for i in range(num_frames):
            ratio = (float(playback_fps)/frame_rate)
            frame = int((start_frame + i) * ratio)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)


            success, img = cap.read()
            if not success:
                img = np.zeros((1080, 1920, 3))
                success = True



            dimensions = img.shape
            if not (img is None or success == False):
                img = Resize(dsize=(self.pre_crop_size, self.pre_crop_size), rel_scale='max').process(img)
                frames.append(img)
                count += 1
            else:
                # TODO: use zero padded image, or fix annotations
                print("WARNING: frame: %d for  filepath %s, start seconds %d, is None" % (i, file_path, start_seconds))
            if count >= num_frames:
                break
        if cache:
            pickle.dump(frames, open(self.cached_path(video_id, start_seconds), "wb"))

        #print("Rreturning frames len {} and size {}".format(len(frames), frames[0].shape))
        return frames, dimensions


    def cached_dir(self):
        return os.path.join(self.data_dir, "cached-%d-%d" % (self.pre_crop_size, self.segment_length))

    def cached_path(self, video_id, start_seconds):
        return os.path.join(self.cached_dir(), "%s-%d.pkl" % (video_id, start_seconds))

    def cached_frames(self, record):
        data_path = self.cached_path(record['video_id'], record['start_seconds'])
        #print("data path is", data_path)
        if os.path.exists(data_path):
            #print("Got a nice cached frame!")
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                return data
        return None

    def video_path(self, video_id):
        return self.data_dir + video_id + ".mp4"

    def download_clip(self, video_id, start_seconds, end_seconds):
        remote_video_path = "https://www.youtube.com/watch?v=" + video_id
        local_path = self.data_dir + 'videos/' + video_id + '-' + str(start_seconds) + '-' + str(end_seconds) + '.mp4'
        start = datetime.timedelta(seconds=start_seconds)
        duration = datetime.timedelta(seconds=(end_seconds - start_seconds))
        command = ("ffmpeg -y -ss %s -i $(youtube-dl -f 22 -g '%s') -t %s -c copy %s > /dev/null 2>&1" % (str(start), remote_video_path, str(duration), local_path))
        process = os.system(command)
        success = process == 0
        return local_path, success

    def zero_frames(self, num_frames=64):
        img_array = []
        for i in range(num_frames):
            img = np.zeros((self.pre_crop_size, self.pre_crop_size, 3))
            img_array.append(img)
        return img_array

    def frames_labels_meta(self, index):
        if not os.path.exists(self.cached_dir()):
            os.mkdir(self.cached_dir())
        record = self.df.iloc[index]
        labels = None
        raw_frames = self.cached_frames(record)
        video_id = record['video_id']
        start_seconds = float(record['start_seconds'])
        end_seconds = float(record['end_seconds'])
        dimensions = [0, 0, 0] #placeholder in case we don't get it from a new inference video
        if not raw_frames:
            #print("Could not find frames in cached")
            video_path = self.video_path(video_id)
            #print("Looking at video_path", video_path, "\n\n")
            if os.path.exists(video_path):
                raw_frames, dimensions = self.get_frames_from_video(video_id, video_path,
                                                    start_seconds, end_seconds)
                #print("EXTRACTED DIMENSIONS", dimensions)
            else:
                local_path, success = self.download_clip(video_id, start_seconds, end_seconds)
                if success:
                    raw_frames = self.get_frames_from_video(video_id, local_path,
                                                    0, start_seconds - end_seconds)
                    print(local_path)
                else:
                    print("WARNING: could not download")
                    raise Exception('Could not download video %s' % video_id)
                    # raw_frames = self.zero_frames()
                    # print("WARNING: created zero frames")


        metadata = record #[record, dimensions]
        return (raw_frames, labels), metadata, dimensions

    def __getitem__(self, index):
        (raw_frames, labels), record, dimensions = self.frames_labels_meta(index)
        if self.mode == 'train':
            raw_frames = augment_raw_frames(raw_frames, method=self.aug_method)
        else:
            raw_frames = augment_raw_frames(raw_frames)
        #print("ACHIEVED RAW FRAMES", len(raw_frames), raw_frames[0].shape)
        input = SurgeryDataset.raw_frames_to_input(raw_frames, method=self.model)


        # Normalize data between range [-1, 1]
        input = ((input / 255.) * 2 - 1).astype(np.float32)

        if self.model == 'BLV':
            if len(input.shape) == 5:
                input = input.transpose([0, 1, 4, 2, 3])
                input = np.reshape(input, (-1, 192, 224, 224))
            else:
                input = input.transpose([0, 3, 1, 2])
                input = np.reshape(input, (-1, 192, 224, 224))

            input = torch.from_numpy(input)
        else:  # Convert to TSN input format
            input = input.transpose([1, 0, 4, 2, 3])
            input = torch.from_numpy(input)
        # Convert to one label per segment.
        #labels = (np.average(labels, axis=1) > 0.5).astype(np.float32)
        #labels = torch.from_numpy(labels)

        labels = np.zeros(len(self.categories), dtype=np.float32)
        labels[self.categories.index(record['category'])] = 1
        labels = torch.from_numpy(labels)
        if torch.cuda.is_available():
            input = input.cuda()
            labels = labels.cuda()
        record_id = [index, dimensions]
        return (input, record_id, labels)

    def __len__(self):
        return len(self.df)
