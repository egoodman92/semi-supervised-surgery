from dataset import SurgeryDataset
import dataset
import click
from train import get_model, get_train_val_data_loaders, run_epoch, save_results
from torch.utils.tensorboard import SummaryWriter
import logger
import time
import torch.nn as nn
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, precision_score


def evaluate(net, batch_size=64, experiment_dir=None, num_videos=10):
    writer = SummaryWriter(experiment_dir)
    anns_path = "data/v0.3.1-anns-5sec.csv"
    train_data_loader, val_data_loader = get_train_val_data_loaders(anns_path, batch_size)
    criterion = nn.BCELoss()
    optimizer = None
    print(set(list(train_data_loader.dataset.df['video_name'])))
    print(set(list(val_data_loader.dataset.df['video_name'])))

    net.eval()
    val_accuracy, y_true, y_score, results = run_epoch(val_data_loader, net, optimizer, criterion, 0)
    df = pd.DataFrame(results)
    print(df)
    save_results(results, '', 0)

    #logger.write_video(writer, net, val_data_loader.dataset, y_score, y_true, num_videos)
    #print("Validation accuracy: %.4f" % val_accuracy)


def cm2df(cm, labels):
    df = pd.DataFrame()
    for i, row_label in enumerate(labels):
        rowdata={}
        for j, col_label in enumerate(labels):
            rowdata[col_label]=cm[i,j]
        df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
    return df[labels]


def eval_results_csv(results_file, model_name):
    df = pd.read_csv(results_file)

    video_ids = list(set(df['video_id']))

    rows = []
    for video_id in video_ids:
        v = df[df['video_id'] == video_id]
        row = {}
        row['Video ID'] = video_id
        row['# Clips'] = len(v)
        row['# Correct'] = sum(v['correct'])
        row['# Incorrect'] = sum(v['correct'] == 0)
        perc_correct = float(row['# Correct']) * 100 / (row['# Incorrect'] + row['# Correct'])
        perc_correct = float("%.1f" % perc_correct)
        row['% Correct'] = perc_correct # ("%.1f%%" % perc_correct)

        url = v.iloc[0]['youtube_url']
        row['URL'] = "[Youtube](%s)" % url

        y_true = list(v['true_label'])
        y_pred = list(v['predicted_label'])
        c = confusion_matrix(y_true, y_pred)
        row['Confusion Matrix'] = c

        rows.append(row)
    videos_df = pd.DataFrame(rows)
    videos_df.reset_index(drop=True, inplace=True)

    count = max(int(len(videos_df) * 0.1), 10)
    top_count_vids = videos_df.sort_values(by=['# Clips'], ascending=False)[0:count]
    top_count_vids = top_count_vids.drop(['Confusion Matrix'], axis=1)

    cols = ['Video ID', '# Clips', '% Correct', '# Correct', '# Incorrect', 'URL']
    top_count_vids = top_count_vids.sort_values(by=['% Correct'], ascending=True)
    print("\n#### Worst predicted videos")
    print(top_count_vids.iloc[:3][cols].to_markdown())

    top_count_vids = top_count_vids.sort_values(by=['% Correct'], ascending=False)
    print("\n#### Best predicted videos")
    print(top_count_vids.iloc[:3][cols].to_markdown())

    incorrect = df[df['correct'] == 0]
    incorrect['youtube_url'] = '[Youtube](' + incorrect['youtube_url'].astype(str) + ')'
    inc_sort = incorrect.sort_values(by=['score_for_true_label'], ascending=True)
    for col in ['score_for_predicted_label', 'score_for_true_label']:
        inc_sort[col] = pd.Series(
        ["{0:.2f}%".format(val * 100) for val in inc_sort[col]], index=inc_sort.index)

    inc_sort.rename(columns={'predicted_label': 'Predicted', 'true_label':'True', 'video_id': 'Video ID',
                                     'score_for_true_label': 'Score for True', 'youtube_url': 'URL',
                                     'score_for_predicted_label': "Score for Predicted"}, inplace=True)
    print("\n#### Worst predicted segments")
    print(inc_sort.iloc[:6][['Video ID', 'True', 'Predicted', 'Score for True', 'Score for Predicted', 'URL']].to_markdown())

    y_true = list(df['true_label'])
    y_pred = list(df['predicted_label'])

    c = confusion_matrix(y_true, y_pred)
    categories = ['cutting', 'tying', 'suturing']
    c_df = cm2df(c, categories)

    print("\n#### Confusion matrix for %d validation segments" % len(df))
    print(c_df.to_markdown())

    perc = float(sum(df['correct']) * 100.0 / len(df))
    acc = ("%.1f%%" % perc)
    res = [{'Model': model_name, 'Accuracy': acc}]
    res_df = pd.DataFrame(res)
    print("\n### Results")
    print(res_df.to_markdown())


@click.command()
@click.option('--exp-dir', default=None)
@click.option('--batch-size', default=2)
@click.option('--model-name', default='TSN')
@click.option('--model-path', default=None)
@click.option('--num-categories', default=3)
@click.option('--num-videos', default=10, help="The number of sample video clips to create")
@click.option('--results_file', default=None)
def start(exp_dir, batch_size, num_categories, model_name, model_path, num_videos, results_file):
    if results_file:
        eval_results_csv(results_file, model_name)
    else:
        SurgeryDataset.categories = dataset.DEFAULT_CATEGORIES[0:num_categories]
        print("Categories: %s" % str(SurgeryDataset.categories))
        net = get_model(num_classes=len(SurgeryDataset.categories), model_name=model_name,
                        model_path=model_path)
        evaluate(net, batch_size, exp_dir, num_videos=num_videos)


if __name__ == '__main__':
    start()