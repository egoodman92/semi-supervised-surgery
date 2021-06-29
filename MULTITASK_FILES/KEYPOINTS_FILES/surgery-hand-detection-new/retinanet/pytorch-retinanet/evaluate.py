import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--model_path', help='Path to model', type=str)

    parser = parser.parse_args(args)

    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    """
    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    """
    weights = parser.model_path
    retinanet = torch.load(weights)
    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    # retinanet.load_state_dict(torch.load(parser.model_path))
    # retinanet = torch.nn.DataParallel(retinanet).cuda()

   #  retinanet.training = False
    retinanet.eval()
    # retinanet.module.freeze_bn()

    # coco_eval.evaluate_coco(dataset_val, retinanet)
    mAP = csv_eval.evaluate(dataset_val, retinanet)


if __name__ == '__main__':
    main()
