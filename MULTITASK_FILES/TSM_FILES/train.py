from dataset import SurgeryDataset
import dataset
from torch.utils.data import DataLoader
import time
import click
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, precision_score, accuracy_score, recall_score
import numpy as np
from torch.nn.functional import softmax
import pandas as pd
import logger
import os
import sys
from model import get_model_name, save_model, save_results, \
    get_model
from barbar import Bar
import git


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_optimizer(name, net, lr=None, weight_decay=1e-10):
    if name == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr)
    return optimizer


def one_hot(x):
    max_idx = np.argmax(x, 1)
    out = np.zeros((max_idx.size, x.shape[1]))
    out[np.arange(max_idx.size), max_idx] = 1
    return out


def batch_results(record_ids, output, labels, dataset):
    records = []
    labels = labels.cpu().numpy()
    output = output.detach().cpu().numpy()
    record_ids = record_ids.cpu().numpy()
    for i in range(len(record_ids)):
        label_index = np.argmax(labels[i])
        prediction = output[i]
        predicted_index = np.argmax(prediction)
        predicted_label = SurgeryDataset.categories[predicted_index]
        score_for_true = prediction[label_index]
        score_for_predicted = prediction[predicted_index]
        label = SurgeryDataset.categories[label_index]
        record_id = int(record_ids[i])
        record = dataset.df.iloc[record_id].to_dict()
        record['score_for_predicted_label'] = score_for_true
        record['score_for_true_label'] = score_for_true
        record['score_for_predicted_label'] = score_for_predicted
        record['predicted_label'] = predicted_label
        record['true_label'] = label
        record['scores'] = prediction
        record['correct'] = int(label == predicted_label)
        records.append(record)
    return records

def run_epoch(data_loader, net, optimizer, criterion, epoch, writer=None):
    all_labels = torch.FloatTensor()
    all_outputs = torch.FloatTensor()
    if torch.cuda.is_available():
        all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()

    loss_value = 0
    loss_value_norm = 0

    mode = 'train' if net.training else 'val'

    results = []
    for index, (data, record_ids, labels) in enumerate(Bar(data_loader)):
        if net.training:
            optimizer.zero_grad()

        num_batches = data.shape[0]


        if get_model_name(net) == 'TSN':
            output = net(data) #then we throw the data through the net!
            #print("OUTPUT1", output.shape)
            output = output.view(-1, int(output.shape[0]/num_batches), output.shape[1])
            #print("OUTPUT2", output.shape)
            output = output.mean(1)
            #print("OUTPUT3", output.shape)
        elif not 'bLVNet' in get_model_name(net):
            output, _ = net(data)
            output = output.view(-1, int(output.shape[0]/num_batches), output.shape[1])
            output = output.mean(1)
        else:
            if len(data.shape) == 5:  # then there are multiple videos to get
                n = data.shape[1]
                data = data.view(-1, data.shape[2], data.shape[3], data.shape[4])
                output = net(data)
                output = output.view(-1, n, output.shape[1])
                output = output.mean(1)
            else:
                output = net(data)

        output = softmax(output)
        #print("SOFTMAX OUTPUT IS", output)

        all_labels = torch.cat((all_labels, labels), 0)
        all_outputs = torch.cat((all_outputs, output.data), 0)

        loss = criterion(output, labels)
        loss_value += loss.data.float()
        loss_value_norm += 1

        #writer.add_scalar('StepLoss/' + mode, loss.data.float(), index)
        if net.training:
            loss.backward()
            optimizer.step()
        results = results + batch_results(record_ids, output, labels, data_loader.dataset)

    y_score = all_outputs.cpu().numpy()
    y_pred = one_hot(y_score)
    y_true = all_labels.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)

    if writer:
        logger.write_accuracy(writer, mode, data_loader, y_pred, y_true, y_score, epoch)

    overall_loss = loss_value / loss_value_norm
    if writer:
        writer.add_scalar('Loss/' + mode, overall_loss, epoch)
    return accuracy, y_true, y_score, results


def make_weights_for_balanced_classes(df, categories):
    weight_per_class = {}
    for i, category in enumerate(categories):
        weight_per_class[category] = len(df)/sum(df['label'] == category)
    weights = [0] * len(df)
    for i in range(len(df)):
        row = df.iloc[i]
        if row['label'] in categories:
            weight = weight_per_class[row['label']]
        else:
            weight = 0
        weights[i] = weight
    print(weight_per_class)
    return weights


def get_train_val_data_loaders(anns_path, batch_size, data_dir='data/', model='BLV', balance=True, k=0, pre_crop_size=256,
                                                                    aug_method='04-20', segment_length=5):
    train_df, val_df = dataset.get_train_val_dfs(anns_path, k=k)

    #print("TRAIN_DF", train_df.shape, train_df.head(2))

    train_dataset = SurgeryDataset(train_df, data_dir=data_dir, mode="train", model=model, balance=balance,
                                   pre_crop_size=pre_crop_size,aug_method=aug_method, segment_length=segment_length)
    print("Train dataset length: %d" % train_dataset.__len__())

    weights = make_weights_for_balanced_classes(train_dataset.df, train_dataset.categories)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False, sampler=sampler)

    val_dataset = SurgeryDataset(val_df, data_dir=data_dir, mode="val", model=model, balance=balance,
                                   pre_crop_size=pre_crop_size, aug_method='val', segment_length=segment_length)
    print("Val dataset length: %d" % val_dataset.__len__())
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)
    return train_data_loader, val_data_loader


def train(net, train_data_loader, val_data_loader, lr=1e-4, batch_size=64,
          max_epochs=int(1e7), eval_every=10,
          opt_name='adam', weight_decay=1e-10,
          exp_path=None):

    writer = SummaryWriter(exp_path)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()

    optimizer = get_optimizer(opt_name, net, lr, weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, mode='min')

    train_losses = []
    val_losses = []
    max_val_accurcy = 0.0

    start = time.time()

    #print("ENTERING TRAINING LOOP!")
    for i in range(1, max_epochs):
        #print("TRAINING", i)
        net.train()
        #print("TRAINING 2")
        train_accuracy, _, _, train_results = run_epoch(train_data_loader, net, optimizer, criterion, i, writer)
        #print("ACCURACY")
        scheduler.step(train_accuracy)

        print("Epoch: %d. Completed epoch in %.5f seconds" % (i, (time.time() - start)))
        start = time.time()

        net.eval()
        val_accuracy, y_score, y_true, val_results = run_epoch(val_data_loader, net, optimizer, criterion, i, writer)

        if val_accuracy > max_val_accurcy:
            max_val_accurcy = val_accuracy
            print("Saving model")
            save_model(net, exp_path, i)

            save_results(train_results, exp_path, i, train=True)
            save_results(val_results, exp_path, i)
            #logger.write_video(writer, val_data_loader.dataset, y_score, y_true)

        if i % eval_every == 0:
            print("\nTrain loss: %.7f" % train_accuracy)
            print("Validation loss: %.7f" % val_accuracy)
            print("Epoch: %d. Completed %d epochs in %.5f seconds" % (i, eval_every, (time.time() - start)))
            start = time.time()

    return train_losses, val_losses


def commit_to_git(exp_name):
    repo = git.Repo(search_parent_directories=True)
    try:
        repo.git.add(u=True)
        repo.git.commit('-m', "Running experiment: %s" % exp_name)
        #repo.remote().push()
    except:
        pass
    return repo.head.object.hexsha


@click.command()
@click.option('--lr', default=1e-4)
@click.option('--batch-size', default=2)
@click.option('--max-epochs', default=21)
@click.option('--eval-every', default=1)
@click.option('--opt-name', default='adam')
@click.option('--weight-decay', default=0.0)
@click.option('--exp-name', default=None)
@click.option('--num-categories', default=4) #somehow this was 3? but in prior analysis, showed four?
@click.option('--model-name', default='TSN')
@click.option('--model-path', default=None)
@click.option('--data-dir', default='data/')
@click.option('--anns-path', default='data/v0.3.1-anns-5sec.csv')
@click.option('--exp-dir', default='/mnt/efs/runs/')
@click.option('--unbalance', is_flag=True)
@click.option('--k', default=0)
@click.option('--pre-crop-size', default=256)
@click.option('--aug-method', default='04-20')
@click.option('--segment-length', default=5)
def start(lr, batch_size, max_epochs, eval_every, opt_name, weight_decay, exp_name, num_categories,
          model_name, model_path, data_dir, anns_path, exp_dir, unbalance, k, pre_crop_size,
          aug_method, segment_length):
    SurgeryDataset.categories = dataset.DEFAULT_CATEGORIES[0:num_categories]
    print("Categories: %s" % str(SurgeryDataset.categories))
    balance = not unbalance

    print("\nGetting model!\n")

    net = get_model(num_classes=len(SurgeryDataset.categories), model_name=model_name, model_path=model_path)
    print("\nModel extracted!\n")
    #net = torch.nn.DataParallel(net)
    if torch.cuda.is_available():
        net = net.cuda()

    train_data_loader, val_data_loader = get_train_val_data_loaders(anns_path, batch_size, data_dir,
                                                                    model_name, balance, k, pre_crop_size,
                                                                    aug_method, segment_length)

    if exp_name is None:
        exp_name = get_model_name(net) + time.strftime("-%I_%M_%p")
    exp_path = os.path.join(exp_dir, exp_name)

    num_train_videos = len(set(train_data_loader.dataset.df['video_id']))
    num_val_videos = len(set(val_data_loader.dataset.df['video_id']))

    # if torch.cuda.is_available():
    #     git_commit = commit_to_git(exp_name)
    # else:
    git_commit = None #= git.Repo(search_parent_directories=True).head.object.hexsha

    config = {'lr': lr, 'batch_size': batch_size, 'opt_name': opt_name, 'weight_decay': weight_decay,
              'exp_name': exp_name, 'num_categories': num_categories, 'model_name': model_name,
              'anns_path': anns_path, 'balanced': (not unbalance), 'commit': git_commit,
              'train_size': train_data_loader.dataset.__len__(), 'val_size': val_data_loader.dataset.__len__(),
              'num_train_videos': num_train_videos, 'num_val_videos': num_val_videos, 'pre_crop_size':pre_crop_size,
              'aug_method': aug_method, 'segment_length': segment_length}
    config_df = pd.DataFrame([config])
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    config_df.to_csv(os.path.join(exp_path, 'config.csv'))
    print("ABOUT TO START TRAINING!")
    train_losses, val_losses = train(net, train_data_loader, val_data_loader,
                                     lr, batch_size, max_epochs, eval_every, opt_name,
                                     weight_decay, exp_path)


if __name__ == '__main__':
    start()
