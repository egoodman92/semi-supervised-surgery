# 
# Re-import of git clone surgery tool detection 
# Edited to remove bugs 
# 
import argparse
import collections
from pathlib import Path
import csv

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, utils as vision_utils
#from torch.utils.tensorboard import SummaryWriter

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer, BalancedSampler
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

MODELS_DIR = str(Path(__file__).resolve().parents[2]) + '/models/'
LOGS_DIR = str(Path(__file__).resolve().parents[2]) + '/logs/'
# pretrained_model = MODELS_DIR + 'coco_resnet_50_map_0_335_state_dict.pt'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--model_name', help='Name to store the trianed model under.')
    parser.add_argument('--log_output', help='Save output to csv file', action='store_true')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--threshold', help='iou threshold to count as detection')
    parser.add_argument('--sampler', help='Type of sampler to use, default aspect ratio sampler.')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--pretrained', help='Path to pretrained model')
    parser.add_argument('--blacken', action='store_true')

    parser = parser.parse_args(args)

    model_name = 'model'
    if parser.model_name is not None:
        model_name = parser.model_name

    learning_rate = 1e-5
    if parser.learning_rate is not None:
        learning_rate = float(parser.learning_rate)

    batch_size = 2
    if parser.batch_size is not None:
        batch_size = int(parser.batch_size)

    threshold = 0.5
    if parser.threshold is not None:
        threshold = float(parser.threshold)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]), augment=parser.augment, blacken=parser.blacken)
        train_acc_set = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Resizer()]), blacken=parser.blacken)

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=batch_size, drop_last=False)
    if parser.sampler is not None and parser.sampler == 'balanced':
        sampler = BalancedSampler(dataset_train, batch_size=batch_size, drop_last=False)

    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    #tb = SummaryWriter('runs/{}'.format(model_name))
    if parser.log_output:
        print('Logging training loss under {}_loss.csv'.format(LOGS_DIR + model_name))
        loss_f = open('{}_loss.csv'.format(LOGS_DIR + model_name), 'w')
        loss_logger = csv.writer(loss_f, delimiter=',')
        loss_logger.writerow(['Epoch', 'Iteration', 'Classification Loss', 'Regression Loss', 'Running Loss'])

        print('Logging training accuracy under {}_train.csv'.format(LOGS_DIR + model_name))
        train_f = open('{}_train.csv'.format(LOGS_DIR + model_name), 'w')
        train_logger = csv.writer(train_f, delimiter=',')
        train_logger.writerow(['Epoch'] + [dataset_train.label_to_name(label) for label in range(dataset_train.num_classes())])

        if parser.csv_val is not None:
            print('Logging validation output under {}_validation.csv'.format(LOGS_DIR + model_name))
            validation_f = open('{}_validation.csv'.format(LOGS_DIR + model_name), 'w')
            val_logger = csv.writer(validation_f, delimiter=',')
            val_logger.writerow(
                ['Epoch'] + [dataset_val.label_to_name(label) for label in range(dataset_val.num_classes())])

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):

            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                if parser.log_output:
                    loss_logger.writerow(['{}'.format(epoch_num), '{}'.format(iter_num), '{:1.5f}'.format(float(classification_loss)),
                                           '{:1.5f}'.format(float(regression_loss)), '{:1.5f}'.format(np.mean(loss_hist))])
                    loss_f.flush()
                    validation_f.flush()
                    train_f.flush()

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            if epoch_num % 10 == 1:
                print('Evaluating dataset on training')
                train_mAP, train_pr = csv_eval.evaluate(train_acc_set, retinanet, iou_threshold=threshold)
                #tb.add_scalars('Training mAP', {train_acc_set.label_to_name(label): train_mAP[label][0] for label in
                #                                range(train_acc_set.num_classes())}, epoch_num)

            print('Evaluating dataset on validation')
            mAP, pr_curve = csv_eval.evaluate(dataset_val, retinanet, iou_threshold=threshold)
            #tb.add_scalars('Validation mAP', {dataset_val.label_to_name(label): mAP[label][0] for label in
            #                                range(dataset_val.num_classes())}, epoch_num)

            if parser.log_output:
                val_logger.writerow([str(epoch_num)] + [mAP[label][0] for label in range(dataset_val.num_classes())])
                if epoch_num % 10 == 1:
                    train_logger.writerow([str(epoch_num)] + [train_mAP[label][0] for label in range(dataset_train.num_classes())])

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, '{}_incomplete.pt'.format(MODELS_DIR + model_name))

    loss_f.close()
    train_f.close()
    validation_f.close()
    retinanet.eval()

    torch.save(retinanet, '{}_final.pt'.format(MODELS_DIR + model_name))

if __name__ == '__main__':
    main()
