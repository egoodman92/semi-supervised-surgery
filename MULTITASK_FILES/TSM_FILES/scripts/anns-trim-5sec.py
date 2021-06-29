"""
If two adjacent annotations have the same label and the end time
of previous annotation is close to start time of next annotation
then combine them into one label.
"""
import pandas as pd
import click

def trim_5sec(ann_path, out_path, reference_anns):
    df = pd.read_csv(ann_path)
    df = df.sort_values(by=['video_id', 'start_seconds'])
    rf = pd.read_csv(reference_anns)

    print("Number of records before dedupe: %d" % len(df))
    df.drop_duplicates(subset=["video_id", "start_seconds", "end_seconds"],
                       keep='first', inplace=True)
    print("Number of records after dedupe: %d" % len(df))

    print("Number of records before dedupe reference: %d" % len(rf))
    rf.drop_duplicates(subset=["video_id", "start_seconds", "end_seconds"],
                       keep='first', inplace=True)
    print("Number of records after dedupe reference: %d" % len(rf))

    count = 0

    rows = []
    for i in range(len(df)):
        curr = df.iloc[i]
        results = rf[(rf['video_id'] == curr['video_id']) &
                     (rf['start_seconds'] >= curr['start_seconds']) &
                     (rf['end_seconds'] <= curr['end_seconds']) &
                     (rf['duration'] < 5) & (rf['duration'] > 0)]
        if len(results) > 0:
            count += 1
            r = results.iloc[0]
            curr['start_seconds'] = r['start_seconds']
            curr['end_seconds'] = r['end_seconds']
            curr['duration'] = curr['end_seconds'] - curr['start_seconds']
        rows.append(curr)

    rf = rf[(rf['duration'] < 5) & (rf['duration'] > 0)]
    for i in range(len(rf)):
        curr = rf.iloc[i]
        results = df[(df['video_id'] == curr['video_id']) &
                     (df['start_seconds'] <= curr['start_seconds']) &
                     (df['end_seconds'] >= curr['end_seconds'])]
        if len(results) == 0:
            rows.append(curr)
            count += 1

    new_df = pd.DataFrame(rows)
    new_df = new_df.sort_values(by=['video_id', 'start_seconds'])

    new_df = new_df.loc[:, ~new_df.columns.str.contains('^Unnamed')]
    new_df = new_df.reset_index()
    new_df = new_df.drop(['index'], axis=1)
    new_df.to_csv(out_path)
    print("Trimmed %d annotations" % count)


@click.command()
@click.option('--ann-path', default='../annotations/v0.5.0-anns-5sec.csv')
@click.option('--out-path', default='../annotations/v0.5.0-anns-5sec-timmed.csv')
@click.option('--reference-anns', default='../annotations/v0.3.1-anns.csv', help='Reference anns containing sub 5 second labels')
def start(ann_path, out_path, reference_anns):
    trim_5sec(ann_path, out_path, reference_anns)


if __name__ == '__main__':
    start()