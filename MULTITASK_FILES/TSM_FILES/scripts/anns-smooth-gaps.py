"""
If two adjacent annotations have the same label and the end time
of previous annotation is close to start time of next annotation
then combine them into one label.
"""
import pandas as pd
import click

def smooth_gaps(ann_path, out_path, threshold):
    df = pd.read_csv(ann_path)
    df = df.sort_values(by=['video_id', 'start_seconds'])
    rows = []
    prev = df.iloc[0]
    for i in range(1, len(df)):
        curr = df.iloc[i]
        diff_in_seconds = curr['start_seconds'] - prev['end_seconds']
        add = True
        new_row = None
        if prev['video_id'] == curr['video_id']:
            if (prev['label'] == 'background') or \
                    diff_in_seconds > 0 and diff_in_seconds <= threshold:
                if prev['label'] != curr['label']:
                    prev['end_seconds'] = curr['start_seconds']
                else:
                    prev['end_seconds'] = curr['end_seconds']
                    add = False
                prev['duration'] = prev['end_seconds'] - prev['start_seconds']
            elif diff_in_seconds > threshold:
                new_row = curr.copy()
                new_row['start_seconds'] = prev['end_seconds']
                new_row['end_seconds'] = curr['start_seconds']
                new_row['label'] = 'background'
                new_row['duration'] = new_row['end_seconds'] - new_row['start_seconds']
                new_row['labeler_1'] = None
                new_row['labeler_2'] = None
                new_row['labeler_3'] = None

        if add is True:
            rows.append(prev)
            prev = curr
        if new_row is not None:
            rows.append(new_row)
    rows.append(prev)

    new_df = pd.DataFrame(rows)
    new_df = new_df.loc[:, ~new_df.columns.str.contains('^Unnamed')]
    new_df = new_df.reset_index()
    new_df = new_df.drop(['index'], axis=1)
    new_df.to_csv(out_path)


@click.command()
@click.option('--ann-path', default='../annotations/v0.5.0-anns.csv')
@click.option('--out-path', default='../annotations/v0.5.0-anns-smoothed.csv')
@click.option('--threshold', default=2, help='Above which should create a background annotation')
def start(ann_path, out_path, threshold):
    smooth_gaps(ann_path, out_path, threshold)


if __name__ == '__main__':
    start()