"""
Inserts background annotations in spaces between annotations.
"""
import pandas as pd
import click

def insert_bgs(ann_path, out_path):
    df = pd.read_csv(ann_path)
    df = df.sort_values(by=['video_id', 'start_seconds'])
    rows = []
    prev = df.iloc[0]
    rows.append(prev)
    for i in range(1, len(df)):
        curr = df.iloc[i]
        if prev['video_id'] == curr['video_id'] and prev['end_seconds'] != curr['start_seconds']:
            new_row = curr.copy()
            new_row['start_seconds'] = prev['end_seconds']
            new_row['end_seconds'] = curr['start_seconds']
            new_row['label'] = 'background'
            new_row['duration'] = new_row['end_seconds'] - new_row['start_seconds']
            new_row['labeler_1'] = None
            new_row['labeler_2'] = None
            new_row['labeler_3'] = None
            rows.append(new_row)
        rows.append(curr)
        prev = curr
    new_df = pd.DataFrame(rows)
    new_df = new_df.loc[:, ~new_df.columns.str.contains('^Unnamed')]
    new_df = new_df.reset_index()
    new_df = new_df.drop(['index'], axis=1)
    new_df.to_csv(out_path)


@click.command()
@click.option('--ann-path', default='../annotations/v0.5.0-anns.csv')
@click.option('--out-path', default='../annotations/v0.5.0-anns2.csv')
def start(ann_path, out_path):
    insert_bgs(ann_path, out_path)


if __name__ == '__main__':
    start()