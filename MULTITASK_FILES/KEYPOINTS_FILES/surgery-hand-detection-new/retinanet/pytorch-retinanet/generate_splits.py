"""
Script to generate new train test splits  
"""

import pandas as pd
import numpy as np
import random
import argparse

total_frames = 1950

def get_df_all(data_path):
    names = ['path', 'x1', 'y1', 'x2', 'y2', 'class_name']
    df_annotations_train = pd.read_csv('{}/annotation_train.csv'.format(data_path), names=names)
    df_annotations_val = pd.read_csv('{}/annotation_val.csv'.format(data_path), names=names)
    df_annotations_test = pd.read_csv('{}/annotation_test.csv'.format(data_path), names=names)
    df_all = pd.concat([df_annotations_train, df_annotations_val, df_annotations_test])
    # Isolate image name in path
    df_all['path'] = df_all['path'].map(lambda x: x.split('/')[-1])
    # Sort alphabetically
    df_all = df_all.sort_values('path').reset_index().drop('index', axis=1)
    # Add video src column
    df_all['video'] = df_all['path'].apply(lambda x: x[-22: -11])
    return df_all


def get_new_splits(df_all, seed, cutoff_train, cutoff_val, path, save=False):
    random.seed(seed)
    groups = [df for _, df in df_all.groupby('video')]
    random.shuffle(groups)
    df_all_s = pd.concat(groups).reset_index(drop=True)
    # Get cutoffs for splits
    last_frame = df_all_s['path'][0]
    last_video = df_all_s['video'][0]
    n_frame = 0
    for i in range(1, df_all_s.shape[0]):
        current_frame = df_all_s['path'][i]
        current_video = df_all_s['video'][i]
        if n_frame > cutoff_train:
            if current_video != last_video:
                cutoff_num_train = i
                cutoff_train = float('inf') # no more
        elif n_frame > cutoff_val and current_video != last_video:
            cutoff_num_val = i
            cutoff_val = float('inf')
        if last_frame != current_frame:
            n_frame += 1
        last_frame = current_frame
        last_video = current_video
    # Finally assign splits
    df_split_train = df_all_s.iloc[0:cutoff_num_train]
    df_split_val = df_all_s.iloc[cutoff_num_train:cutoff_num_val]
    df_split_test = df_all_s.iloc[cutoff_num_val:]
    # Add img as path prefix
    df_split_train['path'] = df_split_train['path'].apply(lambda x: 'img/{}'.format(x))
    df_split_val['path'] = df_split_val['path'].apply(lambda x: 'img/{}'.format(x))
    df_split_test['path'] = df_split_test['path'].apply(lambda x: 'img/{}'.format(x))
    # Remove video
    df_split_train.drop('video', axis=1, inplace=True)
    df_split_val.drop('video', axis=1, inplace=True)
    df_split_test.drop('video', axis=1, inplace=True)
    # Shuffle
    df_split_train = df_split_train.sample(frac=1).reset_index(drop=True)
    df_split_val = df_split_val.sample(frac=1).reset_index(drop=True)
    df_split_test = df_split_test.sample(frac=1).reset_index(drop=True)
    # Save and get train_val split
    df_split_train_val = pd.concat([df_split_train, df_split_val])
    # Save
    if save:
        df_split_train.to_csv('{}/annotation_train-s={}.csv'.format(path, seed), float_format='%.0f', index=False, header=False)
        df_split_val.to_csv('{}/annotation_val-s={}.csv'.format(path, seed), float_format='%.0f', index=False, header=False)
        df_split_test.to_csv('{}/annotation_test-s={}.csv'.format(path, seed), float_format='%.0f', index=False, header=False)
        df_split_train_val.to_csv('{}/annotation_train_val-s={}.csv'.format(path, seed), float_format='%.0f', index=False, header=False)
    return df_split_train, df_split_val, df_split_test, df_split_train_val



def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, help='Random seed for shuffling')
    parser.add_argument('--train_split', type=float, default=0.5, help='Train split ratio')
    parser.add_argument('--val_split', type=float, default=0.2, help='Val split ratio')
    parser.add_argument('--test_split', type=float, default=0.3, help='Test split ratio')
    parser.add_argument('--total_num', type=int, default=1950)
    parser.add_argument('--data_path', type=str, default='../hand-detection-dataset')

    args = parser.parse_args(args)
    
    names = ['path', 'x1', 'y1', 'x2', 'y2', 'class_name']
    
    df_all = get_df_all(args.data_path)  

    cutoff_train = args.total_num * args.train_split  
    cutoff_val = args.total_num * (args.train_split + args.val_split) 

    get_new_splits(df_all, args.seed, cutoff_train, cutoff_val, args.data_path, save=True)
    print('New splits generated!')
    
    
if __name__ == '__main__':
    main() 
