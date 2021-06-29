import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torch.optim as optim

import numpy as np
import random
import sys
import json
import os
import csv
import collections
import argparse
from PIL import Image

import math
import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer, ContextDataset, InferenceDataset, ContextSampler
from retinanet import csv_eval
from retinanet import tracking
from retinanet import model

MODELS_DIR = str(Path(__file__).resolve().parents[2]) + '/models/'
LOGS_DIR = str(Path(__file__).resolve().parents[2]) + '/logs/'

print('CUDA available: {}'.format(torch.cuda.is_available()))
OVERALL_ITERS = 0
BOOTSTRAP_ID = 0

def extract_video_and_frame(image_path):
    filename = image_path.split('/')[-1]
    return '-'.join(filename.split('-')[:-1]), int(filename.split('-')[-1].split('.')[0])

# Function to remove overlapping bounding boxes seen by the  model
def remove_overlaps(prev_annots, new_annots, direction='one', info=None):
    if prev_annots is None or new_annots is None:
        return None
    
    if info is None:
        info = []
    
    overlaps = csv_eval.compute_overlap(prev_annots, new_annots)
    result = np.zeros((0, 5))

    selected_annots = []
    for i, prev_annot in enumerate(prev_annots):
        ordered_overlaps = np.argsort(overlaps[i])[::-1]
        for idx in ordered_overlaps:
            new_annot = new_annots[idx]
            if new_annot[4] != prev_annot[4] or idx in selected_annots:
                continue
            if 'fast-moving' not in info and overlaps[i][idx] < 0.25:
                print('=======Potential false bounding box appeared. Discontinuing in {} direction======'.format(direction))
                return None
            result = np.append(result, new_annot[np.newaxis, :], axis=0)
            # Idea: regress new annot toward the prev annot?
            selected_annots.append(idx)
            break
    if result.shape != prev_annots.shape:
        print('==========Lost a label, discontinuing training in {} direction=========='.format(direction))
        return None
    return result

def train(my_model, my_dataloader, max_iter=8, learning_rate=5e-6, train_loss_logger=None):
    global OVERALL_ITERS
    my_model.training = True
    optimizer = optim.Adam(my_model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=min(max_iter, 200))

    total_iters = 0
    
    my_model.train()
    my_model.module.freeze_bn()
    
    if torch.cuda.is_available():
        my_model = my_model.cuda()
        my_model = torch.nn.DataParallel(my_model).cuda()
    
    for iter_num, data in enumerate(my_dataloader):

        if total_iters > max_iter:
            return

        try:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                classification_loss, regression_loss = my_model([data['img'].cuda().float(), data['annot'].clone()])
            else:
                classification_loss, regression_loss = my_model([data['img'].float(), data['annot'].clone()])

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(my_model.parameters(), 0.1)

            optimizer.step()
            loss_hist.append(float(loss))

            print('Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                total_iters, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            # Log training loss
            if train_loss_logger is not None:
                train_loss_logger.writerow(['{}'.format(OVERALL_ITERS), '{:1.5f}'.format(float(classification_loss)),
                    '{:1.5f}'.format(float(regression_loss)), '{:1.5f}'.format(np.mean(loss_hist))])

            del classification_loss
            del regression_loss
            
        except Exception as e:
            print(e)
            continue
            
        total_iters += 1
        OVERALL_ITERS += 1

def train_initial_frame(my_model, initial_context, initial_dataset, max_iter=40, lr=5e-6):
    print('Training initial frame')
    initial_dataset.add_context(initial_context)
    initial_sampler = ContextSampler(initial_dataset)
    initial_dataloader = DataLoader(initial_dataset, num_workers=4, collate_fn=collater, batch_sampler=initial_sampler)
    
    train(my_model, initial_dataloader, max_iter=max_iter, learning_rate=lr)

# Group video names so that we can train multiple videos per batch (helps to stabailize training)
def group_vids(vid_names, group_size=2):
    return [[vid_names[x % len(vid_names)] for x in range(i, i + group_size)] for i in
            range(0, len(vid_names), group_size)]

def save_annot(out_dict, data, vid_id, mid, frame_difference, labels, img_path='/home/ubuntu/stephen/data/bootstrap_images_from_train'):
    global BOOTSTRAP_ID

    if labels is None:
        labels = {0: 'bovie', 1: 'scalpel', 2: 'forceps', 3: 'needledriver'}

    for i, datum in enumerate(data):
        img, annot = datum
        if img is None or annot is None:
            continue

        direction = 'prev' if i == 0 else 'post'
        filename = '{}-{:09d}_{}_{}.jpg'.format(vid_id, mid, direction, frame_difference)

        Image.fromarray(img, 'RGB').save('{}/{}'.format(img_path, filename))

        # Build annotations
        entry = {
            'object_type': 'image',
            'id': BOOTSTRAP_ID,
            'name': filename,
            'video_id': vid_id,
            'tool_labels': []
        }

        img_height, img_width, _ = img.shape
        for bb in annot:
            entry['tool_labels'].append({
                    'bounding_box_position': {
                        'height': float(bb[3] - bb[1]) / img_height,
                        'width': float(bb[2] - bb[0]) / img_width,
                        'left': float(bb[0]) / img_width,
                        'top': float(bb[1]) / img_height
                    },
                    'category': labels[bb[4]]
                })

        out_dict['data'].append(entry)

        BOOTSTRAP_ID += 1

    return out_dict

def build_segments_from_training(train_f, labels=None):

    durations = {'bovie': 10, 'scalpel': 15, 'forceps': 10, 'needledriver': 10, '': 1}

    if labels is None:
        labels = {'bovie': 0, 'scalpel': 1, 'forceps': 2, 'needledriver': 3, '': -1}

    result = {}

    with open(train_f) as f:
        reader = csv.reader(f)
        for row in reader:
            filename, x1, y1, x2, y2, label = row
            duration = durations[label]
            if label == '':
                bb = None
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                bb = [x1, y1, x2, y2, labels[label]]

            vid_id, frame_number = extract_video_and_frame(filename)

            if vid_id in result:
                central_frames = [entry['middle'] for entry in result[vid_id]]
                if frame_number in central_frames:
                    result[vid_id][central_frames.index(frame_number)]['initial_bbs'].append(bb)
                    if result[vid_id][central_frames.index(frame_number)]['prev'] < duration:
                        result[vid_id][central_frames.index(frame_number)]['prev'] = duration
                        result[vid_id][central_frames.index(frame_number)]['post'] = duration
                else:
                    init_bb = [bb] if bb is not None else []
                    result[vid_id].append({
                        'middle': frame_number,
                        'prev': duration,
                        'post': duration,
                        'initial_bbs': init_bb
                        })
            else:
                init_bb = [bb] if bb is not None else []
                result[vid_id] = [{
                    'middle': frame_number,
                    'prev': duration,
                    'post': duration,
                    'initial_bbs': init_bb          
                }]

        return result

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_train', default='/home/ubuntu/stephen/curis2020/surgery-tool-detection/src/data/train_data.csv', 
        help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', default='/home/ubuntu/stephen/curis2020/surgery-tool-detection/src/data/class_names.csv',
        help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', default='/home/ubuntu/stephen/curis2020/surgery-tool-detection/src/data/test_data.csv',
        help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--model_path', default='/home/ubuntu/stephen/curis2020/surgery-tool-detection/models/cleaned_data_with_negatives_incomplete.pt')
    parser.add_argument('--model_name', default='bootstrap_model')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Bootstrapping learning rate')
    parser.add_argument('--video_batch_size',type=int, default=2)
    parser.add_argument('--log_output', action='store_true')
    parser.add_argument('--focus_tool', help='Train only on segments with a particular tool')
    parser.add_argument('--crystalize', action='store_true')
    parser.add_argument('--memory', type=int, default=None)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--train_from_set', action='store_true')

    parser = parser.parse_args(args)

    # Configure dataset

    dataset = ContextDataset(train_file=parser.csv_train, class_list=parser.csv_classes, memory=parser.memory,
        transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    labels = dataset.labels
    sampler = ContextSampler(dataset)
    dataloader = DataLoader(dataset, num_workers=2, collate_fn=collater, batch_sampler=sampler)
    unnormalize = UnNormalizer()

    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
        transform=transforms.Compose([Normalizer(), Resizer()]))

    # Load model to bootstrap

    pretrained = torch.load(parser.model_path)
    retinanet = model.resnet101(num_classes=dataset.num_classes())
    retinanet.load_state_dict(pretrained.state_dict())

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()

    # Load segment annotations and video metadata

    jf = open('/home/ubuntu/stephen/data/frames_dict.json')
    frame_counts = json.load(jf)

    if parser.train_from_set:
        trained_segments = build_segments_from_training('/home/ubuntu/stephen/curis2020/surgery-tool-detection/src/data/train_data.csv')
    else:
        seg_f = open('/home/ubuntu/stephen/curis2020/surgery-tool-detection/notebooks/bootstrap_train.json')
        trained_segments = json.load(seg_f)

    # Set up loggers
    if parser.log_output:
        print('Logging training loss under {}_loss.csv'.format(LOGS_DIR + parser.model_name))
        loss_f = open('{}_loss.csv'.format(LOGS_DIR + parser.model_name), 'w')
        loss_logger = csv.writer(loss_f, delimiter=',')
        loss_logger.writerow(['Iteration', 'Classification Loss', 'Regression Loss', 'Running Loss'])
        loss_f.flush()

        if parser.csv_val is not None:
            print('Logging validation output under {}_validation.csv'.format(LOGS_DIR + parser.model_name))
            validation_f = open('{}_validation.csv'.format(LOGS_DIR + parser.model_name), 'w')
            val_logger = csv.writer(validation_f, delimiter=',')
            val_logger.writerow(
                ['Iteration'] + [dataset_val.label_to_name(label) for label in range(dataset_val.num_classes())])
            validation_f.flush()
    else:
        loss_logger = None
        val_logger = None

    # Set up outfile so save bootstrapped annotations
    if parser.output_file is not None:
        output_annots = {'data': []}

    # Begin bootstrapping

    vids = list(trained_segments.keys())

    for epoch_num in range(parser.epochs):
        retinanet.training = True
        retinanet.train()
        
        # Group vids so we train on a few videos per batch - helps stabilize training
        random.shuffle(vids)
        groupped_vids = group_vids(vids, group_size=parser.video_batch_size)
        
        for group in groupped_vids:
            data = []
            
            for vid in group:
                try:
                    url = 'https://www.youtube.com/watch?v={}'.format(vid)

                    print('Training batch includes video: {}'.format(vid))
                    entries = trained_segments[vid]

                    if parser.focus_tool is not None: # Remove annotations of extraneous tools
                        updated_entries = []

                        for entry in entries:
                            entry['initial_bbs'] = [bb for bb in entry['initial_bbs'] if labels[bb[4]] == parser.focus_tool]
                            if len(entry['initial_bbs']) > 0:
                                updated_entries.append(entry)

                        entries = updated_entries

                    if len(entries) == 0:
                        continue

                    sequences = [[entry['prev'], entry['middle'], entry['post']] for entry in entries]
                    segments, _ = tracking.get_frames_from_url(url, sequences, orig_frames=frame_counts[vid])
                    data += zip(segments, entries, [vid] * len(entries))
                except Exception as e:
                    print(e)
                    continue
            
            random.shuffle(data)

            # Log initial detection
            if parser.log_output:
                print('Evaluating dataset on validation')
                mAP, pr_curve = csv_eval.evaluate(dataset_val, retinanet)
                val_logger.writerow([str(OVERALL_ITERS)] + [mAP[label][0] for label in range(dataset_val.num_classes())])
                validation_f.flush()
            
            for frames, entry, vid_id in data:

                # Reload the model, since we're just trying to get some bootstrapped data
                print('Reloading model')
                pretrained = torch.load(parser.model_path)
                retinanet = model.resnet101(num_classes=dataset.num_classes())
                retinanet.load_state_dict(pretrained.state_dict())

                if torch.cuda.is_available():
                    retinanet = retinanet.cuda()
                    retinanet = torch.nn.DataParallel(retinanet).cuda()

                frames_prior = entry['prev']
                frames_post = entry['post']
                initial_frame = frames_prior # Index of the central frame
                            
                img = frames[initial_frame]
                
                init_bbs = entry['initial_bbs']
                if len(init_bbs) == 0:
                    init_bbs = np.zeros((0,5))
                else:
                    init_bbs = np.array(init_bbs).astype(np.float32)
                    
                seg_info = None
                if 'info' in entry:
                    seg_info = entry['info']
                    print('Segment has additional info:')
                    print(seg_info)
                
                # Remove when unsupervised
                # initial_dataset = InferenceDataset([img], transform=transforms.Compose([Normalizer(), Resizer()]))
                # dataloader = DataLoader(initial_dataset, collate_fn=collater, batch_size=1)
                # print('Detection on first frame:')
                # tracking.get_bbs(retinanet, dataloader, labels)
                
                # img = tracking.draw_bbs(img, np.array(init_bbs).astype(int), labels)
                # plt.gcf().set_size_inches(15, 10)
                # print('Annotation on first frame:')
                # plt.imshow(img)
                # plt.show()
                # End Remove
                
                bootstrap_seed = [(img, init_bbs)]
                train_initial_frame(retinanet, bootstrap_seed, dataset, lr=parser.learning_rate)
                
                dataset.set_init_frame(bootstrap_seed)
                                      
                past_annots = {'prior': bootstrap_seed[0][1], 'post': bootstrap_seed[0][1]}
                
                # Begin Bootstrap
                frame_difference = 0
                while frame_difference < frames_prior or frame_difference < frames_post:
                    print('Frames trained: {}'.format(frame_difference))
                    frame_difference += 1
                    
                    prev_adjacent = frames[initial_frame - frame_difference] if initial_frame - frame_difference >= 0 else frames[0]
                    post_adjacent = frames[initial_frame + frame_difference] if initial_frame + frame_difference < len(frames) else frames[-1]
                    
                    inf_dataset = InferenceDataset([prev_adjacent, post_adjacent], transform=transforms.Compose([Normalizer(), Resizer()]))
                    inf_dataloader = DataLoader(inf_dataset, shuffle=False, collate_fn=collater, batch_size=1)
                    
                    result = tracking.get_bbs(retinanet, inf_dataloader, labels)
                    
                    # If past a frame boundary, set the annotations to nonetype, so that the dataset disregards it in its 'add' function
                    if frame_difference > frames_prior:
                        print('Reached end of backward training')
                        result[0] = (None, None)
                    if frame_difference > frames_post:
                        print("Reached end of forward training")
                        result[1] = (None, None)
                        
                    result[0] = (result[0][0], remove_overlaps(past_annots['prior'], result[0][1], direction='backward', info=seg_info))
                    result[1] = (result[1][0], remove_overlaps(past_annots['post'], result[1][1], direction='forward', info=seg_info))
                    
                    if result[0][1] is None and result[1][1] is None:
                        # Either we've reached the end or we ran into errors in both directions
                        break

                    # TODO: Save bootstrapped data into dict
                    if parser.output_file is not None and epoch_num < 1:
                        output_annots = save_annot(output_annots, result, vid_id, entry['middle'], frame_difference, labels)
                        jf = open(parser.output_file, 'w')
                        json.dump(output_annots, jf)

                    past_annots['prior'] = result[0][1]
                    past_annots['post'] = result[1][1]
                    
                    # Update the dataloader to include the most recent detections
                    dataset.add_context(result, remember_rate=15 / (frames_prior + frames_post))
                    sampler = ContextSampler(dataset)
                    dataloader = DataLoader(dataset, num_workers=2, collate_fn=collater, batch_sampler=sampler)
                    
                    print('Epoch number: {}'.format(epoch_num))
                    print('Iteration number: {}'.format(OVERALL_ITERS))
                    train(retinanet, dataloader, learning_rate=parser.learning_rate, train_loss_logger=loss_logger)
                    if parser.log_output:
                        loss_f.flush()
            
            # torch.save(retinanet.module, '{}_incomplete.pt'.format(MODELS_DIR + parser.model_name))

        if parser.crystalize:
            new_sampler = ContextSampler(dataset, batch_size=4, crystalize=True)
            new_dataloader = DataLoader(dataset, num_workers=2, collate_fn=collater, batch_sampler=new_sampler)

            for i in range(parser.epochs):
                train(retinanet, new_dataloader, max_iter=len(dataset.learned_context), learning_rate=1e-5)

                if parser.log_output:
                    print('Evaluating dataset on validation')
                    mAP, pr_curve = csv_eval.evaluate(dataset_val, retinanet)
                    val_logger.writerow([str(OVERALL_ITERS)] + [mAP[label][0] for label in range(dataset_val.num_classes())])
                    validation_f.flush()

            break

    if parser.log_output:
        loss_f.close()
        validation_f.close()          

if  __name__ == '__main__':
    main()