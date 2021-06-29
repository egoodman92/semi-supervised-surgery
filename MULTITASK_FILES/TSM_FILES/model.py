import torch
import torch.nn as nn
import pandas as pd
import os
import sys
import torch.hub
repo = 'epic-kitchens/action-models'


resnet3d_path = 'pytorch-resnet3d'
if os.path.exists(resnet3d_path):
    sys.path.insert(0, resnet3d_path)
    from models import resnet
else:
    print('WARNING: clone pytorch-resnet3d')


# run git clone https://github.com/IBM/bLVNet-TAM.git in current folder if missing
blv_path = 'bLVNet-TAM/core/'
if os.path.exists(blv_path):
    sys.path.insert(0, blv_path)
    from models import bLVNet_TAM
else:
    print('WARNING: run "git clone https://github.com/IBM/bLVNet-TAM.git" in current directory to add bLVNet-TAM')


slowfast_path = 'slowfast/slowfast/'
if os.path.exists(slowfast_path):
    sys.path.insert(0, slowfast_path)
    from slowfast.models import build_model
    from slowfast.utils.parser import load_config, parse_args
    import slowfast.utils.checkpoint as cu
else:
    print('WARNING: run "git clone git@github.com:facebookresearch/SlowFast.git" in current directory to add SlowFast')


tsm_path = 'temporal-shift-module'
if os.path.exists(tsm_path):
    sys.path.insert(0, tsm_path)
    from ops.dataset import TSNDataSet
    from ops.models import TSN
    from ops.transforms import *
else:
    print('WARNING: run "git clone git@github.com:mit-han-lab/temporal-shift-module.git" in current directory to addTSM')



def get_model_name(net):
    module = net
    if net.__class__.__name__ == 'DataParallel':
        module = net.module
    return module.__class__.__name__


def save_model(net, experiment_dir, iteration):
    save_dir = os.path.join(experiment_dir, 'saved_models')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    module = net
    if net.__class__.__name__ == 'DataParallel':
        module = net.module
    filename = "%s-%d.pt" % (module.__class__.__name__, iteration)
    save_path = os.path.join(save_dir, filename)
    torch.save(module.state_dict(), save_path)

def load_model(net, model_path):
    saved_model = torch.load(model_path)
    net.load_state_dict(saved_model)


def save_results(results, experiment_dir, iteration, train=False, mode=''):
    save_dir = os.path.join(experiment_dir, 'saved_models')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if train is True:
        mode = 'train'

    filename = "results-%d%s.csv" % (iteration, ('-' + mode if mode is not '' else ''))
    save_path = os.path.join(save_dir, filename)
    df = pd.DataFrame(results)
    df.to_csv(save_path)


def get_model(num_classes, model_name='TSN', freeze_layers=True, model_path=None,):
    class_counts = (125, 352)
    segment_count = 8
    base_model = 'resnet50'
    if model_name == 'BLV':
        backbone_setting = {'dropout': 0.5,
                                'pretrained': True,
                                'alpha': 2,
                                'depth': 101,
                                'beta': 4,
                                'input_channels': 3,
                                'num_classes': 174,
                                'dataset': 'st2stv2',
                                'groups': 64,
                                'imagenet_blnet_pretrained': False,
                                'blending_frames': 3}
        net = bLVNet_TAM(backbone_setting)
    elif model_name == 'slowfast':
        args = parse_args()
        cfg = load_config(args)
        cfg.NUM_GPUS = 1
        cfg.TRAIN.CHECKPOINT_FILE_PATH = "SLOWFAST_4x16_R50.pkl"
        net = build_model(cfg)
    elif model_name == 'TSM':
        net = get_tsm(num_classes)
        # net = torch.hub.load(repo, 'TSM', class_counts, segment_count, 'RGB',
        #                      base_model=base_model,
        #                      pretrained='epic-kitchens')
    elif model_name == 'I3D':
        net = resnet.i3_res50(400)
    else:
        net = torch.hub.load(repo, model_name, class_counts, segment_count, 'RGB',
                         base_model=base_model,
                         pretrained='epic-kitchens', force_reload=True)
    if freeze_layers:
        for param in net.parameters():
            param.requires_grad = False

    if model_name == 'TSN': # or model_name == 'TSM':
        net.fc_verb = torch.nn.Linear(2048, num_classes)
    elif model_name == 'TRN':
        net.consensus.classifiers[0] = torch.nn.Linear(512, num_classes)
    elif model_name == 'BLV':
        net.new_fc = torch.nn.Linear(2048, num_classes)
    elif model_name == 'slowfast':
        net.head.projection = torch.nn.Linear(2304, num_classes)
        net.head.act = None
    elif model_name == 'I3D':
        net.fc =  torch.nn.Linear(2048, num_classes)
    elif model_name == 'TSM':
        net.new_fc = torch.nn.Linear(2048, num_classes)

    if model_path is not None:
        load_model(net, model_path)

    if model_name == 'BLV':
        for param in net.baseline_model.layer4.parameters():
            param.requires_grad = True
    elif model_name in ['TRN', 'TSN', 'TSM']:
        for param in net.base_model.layer4.parameters():
            param.requires_grad = True
    elif model_name == 'I3D':
        for param in net.layer4.parameters():
            param.requires_grad = True

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        net = net.cuda()
    return net


def get_tsm(num_classes=3, pretrain_set='kinetics'):
    if pretrain_set == 'kinetics':
        base_model = "resnet50"
        this_weights = "pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth"
        original_num_classes = 400
        non_local = True
        print("Using kinetics")
    else:
        base_model = "resnet101"
        this_weights = "pretrained/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth"
        # base_model = "resnet50"
        # this_weights = "pretrained/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth"
        original_num_classes = 174
        non_local = False

    modality="RGB"

    segments=8

    consensus_type="avg"
    img_feature_dim=256
    pretrain=True
    is_shift=True
    shift_div=8
    shift_place="blockres"

    print("BUILDING NET")

    net = TSN(original_num_classes, segments, modality,
              base_model=base_model,
              consensus_type=consensus_type,
              img_feature_dim=img_feature_dim,
              pretrain=pretrain,
              is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
              non_local=non_local,)

    print("NET BUILT!")

    checkpoint = torch.load(this_weights)
    checkpoint = checkpoint['state_dict']
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
    replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                    'base_model.classifier.bias': 'new_fc.bias',
                    }
    for k, v in replace_dict.items():
        if k in base_dict:
            base_dict[v] = base_dict.pop(k)

    net.load_state_dict(base_dict)
    #
    # for param in net.parameters():
    #     param.requires_grad = False
    #
    # for param in net.base_model.layer4.parameters():
    #     param.requires_grad = True

    net.new_fc = torch.nn.Linear(2048, num_classes)
    return net


if __name__ == '__main__':

    net = get_model(3, 'slowfast', False)
    print("Loaded model")
    print(net)

