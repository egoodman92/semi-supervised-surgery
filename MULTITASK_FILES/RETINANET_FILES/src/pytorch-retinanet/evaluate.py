import argparse

from retinanet import csv_eval
from retinanet.dataloader import CSVDataset, Resizer, Normalizer, Augmenter
from torchvision import transforms
import torch

def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path')
    parser.add_argument('--val_path')
    parser.add_argument('--train_path')
    parser.add_argument('--class_path')
    parser.add_argument('--threshold')

    args, leftover = parser.parse_known_args()

    model_path = 'model'
    if args.model_path is not None:
        model_path = args.model_path

    threshold = 0.5
    if args.threshold is not None:
        threshold = float(args.threshold)
        
    dataset_train = CSVDataset(train_file=args.train_path, class_list=args.class_path,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    dataset_val = CSVDataset(train_file=args.val_path, class_list=args.class_path,
                             transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    
    retinanet = torch.load(args.model_path)

    print('Evaluating train:')
    csv_eval.evaluate(dataset_train, retinanet, iou_threshold=threshold)

    print('Evaluating val:')
    csv_eval.evaluate(dataset_val, retinanet, iou_threshold=threshold)

    
if __name__ == '__main__':
    main()
