from dataset import SurgeryDataset
import dataset
import click
from train import get_train_val_data_loaders, run_epoch
from model import get_model_name, save_model, save_results, get_model
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import cv2
import utils


def get_test_data_loaders(segments_df, batch_size, data_dir='data/', model='BLV', balance=True, pre_crop_size=256,
                                                                    aug_method='val', segment_length=5):
    df = segments_df.sort_values(by=['video_id', 'start_seconds'])
    test_dataset = SurgeryDataset(df, data_dir=data_dir, mode='test', model=model, balance=balance,
                                   pre_crop_size=pre_crop_size, aug_method=aug_method, segment_length=segment_length)
    print("Number of segments: %d" % test_dataset.__len__())
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)
    return test_data_loader


def evaluate(net, video_ids, anns_path, batch_size=64, experiment_dir=None, use_anns=False, balance=False):

    segments_df = dataframe_from_video_ids(video_ids, anns_path, use_anns=use_anns)
    test_data_loader = get_test_data_loaders(segments_df, batch_size, balance=balance)

    criterion = nn.BCELoss()
    optimizer = None

    net.eval()
    val_accuracy, y_true, y_score, results = run_epoch(test_data_loader, net, optimizer, criterion, 0)
    df = pd.DataFrame(results)
    df = df.drop_duplicates(subset=['video_id', 'start_seconds'], keep=False)
    print(df)
    save_results(results, experiment_dir, 0, mode='inference')


def get_video_path(video_id):
    return "data/videos/" + video_id + ".mp4"


def get_video_duration(filename):
    video = cv2.VideoCapture(filename)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    return (frame_count/fps)


def dataframe_from_video_ids(video_ids, anns_path, segment_lenth=5, use_anns=False):
    segments = []
    anns = pd.read_csv(anns_path)
    anns['label'] = anns['labeler_2']
    anns.loc[anns['label'] == 'none', 'label'] = 'background'

    if use_anns:
        anns = anns[~anns['labeler_2'].isnull()]
        anns['label'] = anns['labeler_2']
        anns['category'] = anns['label']
        anns['video_name'] = anns['video_id']
        return anns[anns['video_id'].isin(video_ids)]

    for video_id in video_ids:
        video_path = get_video_path(video_id)
        duration = get_video_duration(video_path)
        for i in range(int(duration/5)-1):
            start_seconds = segment_lenth * i
            end_seconds = segment_lenth * (i + 1)
            duration = end_seconds - start_seconds
            ann = anns[(anns['video_id'] == video_id)]
            ann = ann[(ann['start_seconds'] >= start_seconds) & (ann['start_seconds'] < end_seconds) | (ann['end_seconds'] > start_seconds) & (ann['end_seconds'] <= end_seconds)]

            labels = list(set(ann['label']))
            if len(labels) == 1:
                label = labels[0]
            elif len(labels) == 2:
                if 'background' in labels:
                    labels.remove('background')
                    label = labels[0]
                else:
                    label = 'abstain'
            else:
                label = 'abstain'

            segment = {'video_id': video_id,
                       'duration': duration,
                       'start_seconds': segment_lenth * i,
                       'end_seconds': segment_lenth * (i + 1),
                       'label': label
                       }
            segments.append(segment)
    df = pd.DataFrame(segments)
    df['category'] = df['label']
    return df


@click.command()
@click.option('--exp-dir', default=None)
@click.option('--batch-size', default=2)
@click.option('--model-name', default='TSN')
@click.option('--model-path', default=None)
@click.option('--num-categories', default=4)
@click.option('--video-ids', default=None)
@click.option('--anns-path', default='data/v0.4.0-anns-bg-5sec.csv')
@click.option('--use-anns', is_flag=True)
@click.option('--balance', is_flag=True)
def start(exp_dir, batch_size, model_name, model_path, num_categories, video_ids, anns_path, use_anns, balance):
    if exp_dir is None:
        exp_dir = "/".join(model_path.split("/")[0:-2])
        print(exp_dir)

    if video_ids is None:
        exps = utils.get_experiments()
        exp_name = exp_dir.split('/')[-1]
        exp = exps[exps['experiment_name'] == exp_name].iloc[0]
        val_df = pd.read_csv(exp['results_file'])
        video_ids = list(set(val_df['video_id']))
        print(video_ids)
    else:
        video_ids = video_ids.split(',')

    SurgeryDataset.categories = dataset.DEFAULT_CATEGORIES[0:num_categories]
    print("Categories: %s" % str(SurgeryDataset.categories))
    net = get_model(num_classes=len(SurgeryDataset.categories), model_name=model_name,
                        model_path=model_path)
    evaluate(net, video_ids, anns_path, batch_size, exp_dir, use_anns=use_anns, balance=balance)


if __name__ == '__main__':
    start()


