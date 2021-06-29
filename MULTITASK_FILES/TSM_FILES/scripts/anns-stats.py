import pandas as pd
import click
import statistics


def stats(ann_path):
    df = pd.read_csv(ann_path)
    df = df.sort_values(by=['video_id', 'start_seconds'])

    if not 'label' in df.columns:
        df['label'] = df['labeler_2']
        df.loc[df['label'] == 'none', 'label'] = 'background'
        df.loc[df['labeler_2'].isnull(), 'label'] = df['labeler_1']
        df.loc[~df['labeler_3'].isnull(), 'label'] = df['labeler_3']
    print("Number of records before dedupe: %d" % len(df))
    df.drop_duplicates(subset=["video_id", "start_seconds", "end_seconds", "label"],
                       keep='first', inplace=True)
    print("Number of records after dedupe: %d" % len(df))

    label_counts = {}
    label_seconds = {}
    label_minutes = {}
    label_avg_length = {}
    label_median_length = {}
    labels = ['cutting', 'tying', 'suturing', 'background', 'abstain']
    #labels = list(set(df['label']))

    for label in labels:
        label_counts[label] = len(df[df['label'] == label])
        label_seconds[label] = sum(df[df['label'] == label]['duration'])
        label_minutes[label] = round(sum(df[df['label'] == label]['duration']) / 60, 1)
        label_avg_length[label] = round(label_seconds[label] / label_counts[label], 1)
        label_median_length[label] = statistics.median(list(df[df['label'] == label]['duration']))

    video_ids = list(set(df['video_id']))
    window_length = 10
    label_changes_in_10sec_window = {}
    labels_sandwiched = {}
    for video_id in video_ids:
        stats_df = df[df['video_id'] == video_id]
        duration = int(stats_df.iloc[len(stats_df)-1]['end_seconds'])
        windows = int(duration / window_length)
        for i in range(windows):
            start = i * window_length
            end = (i+1) * window_length

            results = stats_df[(stats_df['start_seconds'] >= start) & (stats_df['start_seconds'] < end) |
                        (stats_df['end_seconds'] > start) & (stats_df['end_seconds'] <= end)]
            results = results[~results['label'].isin(['background', 'abstain'])]
            labels = list(results['label'])
            label_changes = 0
            middle_label = None
            if len(labels) > 0:
                last_label = labels[0]
                for l in labels[1:]:
                    if l != last_label:
                        label_changes += 1
                        if label_changes == 1:
                            middle_label = l
                    last_label = l
            if middle_label and label_changes > 1:
                if not middle_label in labels_sandwiched:
                    labels_sandwiched[middle_label] = 0
                labels_sandwiched[middle_label] += 1

            if not label_changes in label_changes_in_10sec_window:
                label_changes_in_10sec_window[label_changes] = 0
            label_changes_in_10sec_window[label_changes] += 1

    label_changes_perc = {}
    for k, v in label_changes_in_10sec_window.items():
        label_changes_perc[k] = v / sum(label_changes_in_10sec_window.values())

    all_stats = {}
    all_stats['# Annotations'] = label_counts
    all_stats['Total Minutes'] = label_minutes
    all_stats['Avg Length (s)'] = label_avg_length
    all_stats['Median Length (s)'] = label_median_length
    all_stats['# Sandwiched in 10s Window'] = labels_sandwiched

    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.transpose()
    print(stats_df)
    return stats_df


@click.command()
@click.option('--ann-path', default='../annotations/v0.5.0-anns-smoothed.csv')
def start(ann_path):
    stats(ann_path)


if __name__ == '__main__':
    start()
