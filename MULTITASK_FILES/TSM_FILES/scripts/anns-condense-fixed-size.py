import pandas as pd
import click

def condense(ann_path, out_path, segment_length):
    df = pd.read_csv(ann_path)
    df = df.sort_values(by=['video_id', 'start_seconds'])

    if not 'label' in df.columns:
        df['label'] = df['labeler_2']
        df.loc[df['label'] == 'none', 'label'] = 'background'
        df.loc[df['labeler_2'].isnull(), 'label'] = df['labeler_1']
        df.loc[~df['labeler_3'].isnull(), 'label'] = df['labeler_3']

    print("Number of records before dedupe: %d" % len(df))
    df.drop_duplicates(subset=["video_id", "start_seconds", "end_seconds"],
                       keep='first', inplace=True)
    print("Number of records after dedupe: %d" % len(df))

    parent_starts = []
    parent_ends = []
    durations = []
    end = False
    index = 0
    while not end:
        row = df.iloc[index]
        parent_start = row['start_seconds']
        next_row_index = index + 1
        last_start = row['start_seconds']
        last_duration = row['duration']
        while next_row_index < len(df) and df.iloc[next_row_index]['start_seconds'] == last_start + last_duration and \
                df.iloc[next_row_index]['label'] == row['label']:
            last_start = df.iloc[next_row_index]['start_seconds']
            last_duration = df.iloc[next_row_index]['duration']

            tmp = df.iloc[next_row_index]
            if tmp['video_id'] == 'CfFrwiwgniU':
                if tmp['start_seconds'] == 336.0 or tmp['start_seconds'] == 337.0:
                    print(tmp)

            next_row_index += 1

        last_row = df.iloc[next_row_index - 1]
        parent_end = last_row['end_seconds']
        for i in range(index, next_row_index):
            parent_starts.append(parent_start)
            parent_ends.append(parent_end)
            durations.append(parent_end - parent_start)
        index = next_row_index
        if index >= len(df):
            end = True

    df['parent_start'] = parent_starts
    df['parent_end'] = parent_ends
    df['duration'] = durations

    df.drop_duplicates(subset=["video_id", "parent_start", "parent_end", "label"], keep='first', inplace=True)

    df = df.drop(['start_seconds', 'end_seconds'], axis=1)
    df = df.drop(['start_frame', 'end_frame'], axis=1)
    df.rename(columns={'parent_start': 'start_seconds'}, inplace=True)
    df.rename(columns={'parent_end': 'end_seconds'}, inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.reset_index()
    df = df.drop(['index'], axis=1)

    df.to_csv(out_path)


@click.command()
@click.option('--ann-path', default='../annotations/v0.5.0-anns-5sec.csv')
@click.option('--out-path', default='../annotations/v0.5.0-anns.csv')
@click.option('--segment-length', default=5)
def start(ann_path, out_path, segment_length):
    condense(ann_path, out_path, segment_length)


if __name__ == '__main__':
    start()