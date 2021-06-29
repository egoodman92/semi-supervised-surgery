import pandas as pd


df = pd.read_csv('../data/v0.3.1-anns-bg-5sec.csv')
df = df[~df['labeler_2'].isnull()]

def calculate_unidirectional(df):
    d = {}
    prev = df.iloc[0]

    for i in range(1, len(df)):
        r = df.iloc[i]
        l = r['labeler_2']
        if not l in d:
            d[l] = {'count': 0}
        prev_l = prev['labeler_2']
        if not prev_l in d[l]:
            d[l][prev_l] = 0
        d[l][prev_l] += 1
        d[l]['count'] += 1
        prev = r
    return d

def calculate_bidirectional(df):
    d = {}
    prev = df.iloc[0]

    for i in range(1, len(df)-1):
        r = df.iloc[i]
        n = df.iloc[i+1]
        l = prev['labeler_2'] + '-' + n['labeler_2']
        if not l in d:
            d[l] = {'count': 0}

        current_label = r['labeler_2']
        if not current_label in d[l]:
            d[l][current_label] = 0
        d[l][current_label] += 1
        d[l]['count'] += 1
        prev = r
    return d



# d = calculate_unidirectional(df)
# df2 = pd.DataFrame(d)
# print(df2)
# df2.to_csv('temporal-dependence-uni-directional.csv')

d = calculate_bidirectional(df)
df2 = pd.DataFrame(d)
print(df2)
df2.to_csv('temporal-dependence-bi-directional.csv')
