import json
import os
import pandas as pd
import torch


def videos_by_quality(allowed_qualities):
    dir_name = os.path.join("annotations", "marv_anns")
    files = []
    for i in range(0, 13):
        if i != 9:
            files.append(str(i) + ".json")
    video_qualities = {}
    video_quality_options = ['good', 'okay', 'bad']

    for file in files:
        filepath = os.path.join(dir_name, file)
        with open(filepath, 'r') as f:
            json_string = f.read()
            json_data = json.loads(json_string)
        data = json_data['data']

        for d in data:
            if 'quality' in d:
                video_id = d['id']
                if not video_id in video_qualities:
                    video_qualities[video_id] = []
                if d['quality'] in video_quality_options:
                    video_qualities[video_id].append(d['quality'])
    video_ids = []
    avg_video_quality = {}
    for video_id, qualities in video_qualities.items():
        avg_quality = max(qualities)
        avg_video_quality[video_id] = max(qualities)
        if avg_quality in allowed_qualities:
            video_ids.append(video_id)

    return video_ids


def segments_by_nonexpert_agreement(df):
    filepath = os.path.join("annotations", "nonexpert-action-labels", "1-125.csv")
    n = pd.read_csv(filepath)
    n['seg_id'] = n['video_id'] + "-" + n.astype(str)['start_seconds'] + "-" + n.astype(str)['end_seconds']
    m = n[n['label'] == n['expert_label']]
    mismatch = n[n['label'] != n['expert_label']]
    df['seg_id'] = df['video_name'] + "-" + df.astype(str)['start_seconds'] + "-" + df.astype(str)['end_seconds']
    return df[~df['seg_id'].isin(mismatch['seg_id'])]


def segments_by_length(df, max_length):
    return df[df['duration'] <= max_length]



def get_accuracy(results_file):
    df = pd.read_csv(results_file)
    return sum(df['correct']) / float(len(df))


def get_experiments(runs_path='/mnt/efs/runs/'):
    experiments = [x for x in next(os.walk(runs_path))[1]]
    experiments_info = []
    for exp in experiments:
        dir_path = os.path.join(runs_path, exp)
        config_path = os.path.join(dir_path, 'config.csv')
        if os.path.exists(config_path):
            config = pd.read_csv(config_path)
        else:
            config = None
        model_name = config.iloc[0]['model_name'] if config is not None else None
        saved_models_path = os.path.join(dir_path, 'saved_models')
        if os.path.exists(saved_models_path):
            results = [f for f in os.listdir(saved_models_path) if os.path.isfile(os.path.join(saved_models_path, f))]
            best_results_file = None
            best_model_path = None
            max_acc = 0
            for results_file in results:
                results_path = os.path.join(saved_models_path, results_file)
                if 'result' in results_file and not 'train' in results_file and not 'test' in results_file:
                    acc = get_accuracy(results_path)
                    if acc > max_acc:
                        max_acc = acc
                        best_results_file = results_path
                        best_model_path = results_path.replace(".csv", '.pt')
                        if os.path.exists(best_model_path.replace('results', str(model_name))):
                            best_model_path = best_model_path.replace('results', str(model_name))
                        else:
                            best_model_path = best_model_path.replace('results-', 'TSN-')

            if best_results_file:
                info = {'experiment_name': exp,
                        'results_file': best_results_file,
                        'accuracy': max_acc,
                        'best_model_path': best_model_path,
                        'model_name': model_name
                        }
                experiments_info.append(info)
    df = pd.DataFrame(experiments_info)
    df = df.sort_values(by=['accuracy'], ascending=False)
    return df


def best_model_path(model_name, model_class_name):
    d = get_experiments()
    #d2 = d[d['model_name'] == model_name]
    d2 = d[d['experiment_name'].str.lower().str.contains(model_name.lower())]
    if len(d2) == 0:
        d2 = d[d['model_name'] == model_class_name]
    return d2.iloc[0]['best_model_path']