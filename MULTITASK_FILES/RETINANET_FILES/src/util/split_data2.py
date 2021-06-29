"""
Splits data into training, test, and validation sets
"""
import pandas as pd
from pathlib import Path

def extract_video(image_path):
    filename = image_path.split('/')[-1]
    return '-'.join(filename.split('-')[:-1])

pd.set_option('precision', 0)

DATA_DIR = str(Path(__file__).resolve().parents[1]) + "/data"
DEFAULT_RAW_DATA = DATA_DIR + "/raw_data2.csv" #
DEFAULT_TRAIN = DATA_DIR + "/train_data2.csv" #
DEFAULT_VAL = DATA_DIR + "/val_data2.csv" #
DEFAULT_TEST = DATA_DIR + "/test_data2.csv" #
TEST_SPLIT = DATA_DIR + "/test_split.csv"
TRAIN_SPLIT = DATA_DIR + "/train_split.csv"

df = pd.read_csv(DEFAULT_RAW_DATA, dtype=str, header=None)

train_vids = pd.read_csv(TRAIN_SPLIT, dtype=str)
# train_vids = train_vids.sample(frac=0.25) # Uncomment to play with a smaller toy dataset
val_vids = train_vids.sample(frac=0.15)
train_vids = train_vids.drop(val_vids.index)
test_vids = pd.read_csv(TEST_SPLIT, dtype=str)
vid_ids = df[0].map(extract_video)

train_mask = [vid_id in train_vids["video_id"].tolist() for vid_id in vid_ids]
train = df[train_mask]
val_mask = [vid_id in val_vids["video_id"].tolist() for vid_id in vid_ids]
val = df[val_mask]
test_mask = [vid_id in test_vids["video_id"].tolist() for vid_id in vid_ids]
test = df[test_mask]

train.to_csv(DEFAULT_TRAIN, index=False, header=False)
val.to_csv(DEFAULT_VAL, index=False, header=False)
test.to_csv(DEFAULT_TEST, index=False, header=False)

