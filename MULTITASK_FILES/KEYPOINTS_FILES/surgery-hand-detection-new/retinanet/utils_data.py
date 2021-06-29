"""
Scripts / points that might be helpful for data processing
"""

# Save dataset 
# Also if  annotation coordinates are not int:
def save_csv(df, fpath):
    df.to_csv(fpath, float_format='%.0f', index=False, header=False)


# If setting annotations to hands only
def change_class_name(annotation_path, class_name):
    names = ['path', 'x1', 'y1', 'x2', 'y2', 'class_name']
    df = pd.read_csv(annotation_path, names=names)
    df['class_name'] = class_name
    df['class_name'] = df.apply(lambda r: np.nan if np.isnan(r['y2']) else r['class_name'], axis=1)
    return df

# Join train and val datasets 
def get_train_val(annotations, fpath):
    # :annotations: list of annotation csv's, e.g. [annotation_train.csv, annotation_val.csv]
    names = ['path', 'x1', 'y1', 'x2', 'y2', 'class_name']
    dfs = []
    for f in annotations:
        dfs.append(pd.read_csv(f, names=names))
    df_combined = pd.concat(dfs)
    df_combined.to_csv(fpath, float_format='%.0f', index=False, header=False)


# Generate new train / val / test split  

## Generate concatenated dataframe with all annotations
def get_df_all():
    names = ['path', 'x1', 'y1', 'x2', 'y2', 'class_name']
    df_annotations_train = pd.read_csv('annotation_train.csv', names=names)
    df_annotations_val = pd.read_csv('annotation_val.csv', names=names)
    df_annotations_test = pd.read_csv('annotation_test.csv', names=names)
    df_all = pd.concat([df_annotations_train, df_annotations_val, df_annotations_test])
    # Isolate image name in path
    df_all['path'] = df_all['path'].map(lambda x: x.split('/')[-1])
    # Sort alphabetically
    df_all = df_all.sort_values('path').reset_index().drop('index', axis=1)
    # Add video src column
    df_all['video'] = df_all['path'].apply(lambda x: x[-22: -11])
    return df_all

## Return list of different split seed cutoffs  
def list_splits(df_all, num_seeds):
    for rs in range(20):
        random.seed(rs)
        groups = [df for _, df in df_all.groupby('video')]
        random.shuffle(groups)
        df_all_s = pd.concat(groups).reset_index(drop=True)
        # Set video-wise cutoffs
        cutoff_train = 975
        cutoff_val = 975 + 390
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
        print(rs, cutoff_num_train, cutoff_num_val)

def get_new_splits(df_all, seed, cutoff_train=975, cutoff_val=975 + 390, save=False):
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
        df_split_train.to_csv('annotation_train_s={}.csv'.format(seed), float_format='%.0f', index=False, header=False)
        df_split_val.to_csv('annotation_val_s={}.csv'.format(seed), float_format='%.0f', index=False, header=False)
        df_split_test.to_csv('annotation_test_s={}.csv'.format(seed), float_format='%.0f', index=False, header=False)
        df_split_train_val.to_csv('annotation_train_val_s={}.csv'.format(seed), float_format='%.0f', index=False, header=False) 
    return df_split_train, df_split_val, df_split_test, df_split_train_val

  
        
