"""
Script that copies datasets to the current working directory ('pytorch_retinanet/') and edits the data paths accordingly.
"""

import pandas as pd

dataset_dir = '../hand-detection-dataset'

oxford = False
old_surgery = False

if oxford:
    dataset_dir = '../oxford-hand-detection-dataset'

if old_surgery:
    dataset_dir = '../old-surgery-hand-detection-dataset'

# csv_fnames = ['annotation_train.csv', 'annotation_val.csv', 'annotation_test.csv']
csv_fnames = ['annotation_train-s=15.csv', 'annotation_val-s=15.csv', 'annotation_test-s=15.csv', 'annotation_train_val-s=15.csv']
# csv_fnames = ['annotation_test.csv', 'annotation_train_val.csv']
headers = ['path', 'x1', 'y1', 'x2', 'y2', 'class_name']

for csv in csv_fnames:
    try:
        df = pd.read_csv('{}/{}'.format(dataset_dir, csv), names=headers)
        df['path'] = df['path'].apply(lambda x: '{}/{}'.format(dataset_dir, x))
        if oxford:
            df.to_csv('oxford-{}'.format(csv), float_format='%.0f', index=False, header=False)
        elif old_surgery:
            df.to_csv('old_surgery-{}'.format(csv), float_format='%.0f', index=False, header=False)
        else:
            df.to_csv(csv, float_format='%.0f', index=False, header=False)
    except FileNotFoundError:
        print('{}/{} not found'.format(dataset_dir, csv))


