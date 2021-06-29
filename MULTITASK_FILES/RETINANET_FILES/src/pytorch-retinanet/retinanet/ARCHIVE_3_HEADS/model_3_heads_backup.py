print("INITALIZATIN CRAZY 3-HEADED MODEL!")

import sys
#these next three lines are to important important TSM module functionality
sys.path.append("/home/egoodman/multitaskmodel/MULTITASK_FILES/TSM_FILES/temporal-shift-module/")
from ops.basic_ops import ConsensusModule
from ops.transforms import *


import torch.nn as nn
import torch
from torch.nn.functional import softmax

import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]



class ActionModel(nn.Module):
    def __init__(self, num_actions=4):
        super(ActionModel, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Dropout(p=0.8)
        self.new_fc = nn.Linear(in_features=2048, out_features=4, bias=True)
        self.consensus = ConsensusModule("avg")

    def forward(self, x):
        x = self.avgpool(x)
        x = self.new_fc(x.squeeze())

        #this is so we don't train action head with small detections, i.e. batch size 2
        #if x.shape[0] < 32:
        #    return torch.zeros([4, 4])

        x = x.view((-1, 8) + x.size()[1:])
        out = self.consensus(x)
        return out.squeeze(1)



class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.num_batches = 1
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])


        ######CREATE ACTION HEAD#####
        print("Creating action head")
        self.action_model = ActionModel()
        self.action_loss_criterion = nn.BCELoss()
        #############################
        #############################

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        ######################################################
        self.regressionModel_hands = RegressionModel(256)
        self.classificationModel_hands = ClassificationModel(256, num_classes=num_classes)
        ######################################################


        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)


        self.classificationModel_hands.output.weight.data.fill_(0)
        self.classificationModel_hands.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel_hands.output.weight.data.fill_(0)
        self.regressionModel_hands.output.bias.data.fill_(0)

        #self.freeze_bn()



    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(ResNet, self).train(mode)
        count = 0
        if mode:
            print("Freezing BN layers as we enter training mode!")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= 2:
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False




    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            #img_batch, annotations = inputs
            img_batch, annotations, action_label = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        regression_tools = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification_tools = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        regression_hands = torch.cat([self.regressionModel_hands(feature) for feature in features], dim=1)
        classification_hands = torch.cat([self.classificationModel_hands(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:


            ####WORKING WITH ACTION HEAD ####
            if x4.shape[0] >= 8:
                action_logits = self.action_model(x4)
                action_logits = action_logits.view(-1, int(action_logits.shape[0]/self.num_batches), action_logits.shape[1])
                action_logits = action_logits.mean(1)
                action_logits = softmax(action_logits)
                #print("LOGITS ARE", action_logits)
            else:
                action_logits = torch.tensor([0, 0, 0, 0])
                #print("ACTION LOGITS ARE", action_logits)
            #################################


            hand_annotations = annotations.clone()
            for b in range(hand_annotations.shape[0]):
                for a in range(hand_annotations.shape[1]):
                    if int(hand_annotations[b, a, 4]) in {0, 1, 2}:
                        hand_annotations[b, a, :] = -1

            tool_annotations = annotations.clone()
            for b in range(tool_annotations.shape[0]):
                for a in range(tool_annotations.shape[1]):
                    if int(tool_annotations[b, a, 4]) in {3}:
                        tool_annotations[b, a, :] = -1


            class_loss_hand, reg_loss_hand = self.focalLoss(classification_hands, regression_hands, anchors, hand_annotations)
            class_loss_tool, reg_loss_tool = self.focalLoss(classification_tools, regression_tools, anchors, tool_annotations)
            return 0.5*(class_loss_hand + class_loss_tool), 0.5*(reg_loss_hand + reg_loss_tool), action_logits

        else:

            #this if/else statement is inference on the actions
            if img_batch.shape[0] > 1:
                #this is what we do if we're evaluating the action head
                ####WORKING WITH ACTION HEAD ####
                action_logits = self.action_model(x4)
                action_logits = action_logits.view(-1, int(action_logits.shape[0]/self.num_batches), action_logits.shape[1])
                action_logits = action_logits.mean(1)
                action_logits = softmax(action_logits)
                #################################

            else:
                #this is what we'll do if we're evaluating the detection head on a single frame
                action_logits = None

            transformed_anchors = self.regressBoxes(anchors, regression_tools)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            all_nms_scores = []
            all_nms_classes = []
            all_transformed_anchors = []

            for index in range(classification_tools.shape[0]): #iterate 1

                scores_tools = torch.max(classification_tools[index, :, :].unsqueeze(0), dim=2, keepdim=True)[0]
                scores_over_thresh_tools = (scores_tools.squeeze() > 0.05) #checks scores for each image

                if scores_over_thresh_tools.sum() == 0:
                    all_nms_scores.append(torch.zeros(1, 1))
                    all_nms_classes.append(torch.zeros(1, 1))
                    all_transformed_anchors.append(torch.zeros(1, 1))
                    continue

                image_classification_tools = classification_tools[index, scores_over_thresh_tools, :]
                image_transformed_anchors_tools = transformed_anchors[index, scores_over_thresh_tools, :]
                scores_tools = scores_tools.squeeze()[scores_over_thresh_tools]

                anchors_nms_idx_tools = nms(image_transformed_anchors_tools, scores_tools, 0.25)

                nms_scores_tools, nms_class_tools = image_classification_tools[anchors_nms_idx_tools, :].max(dim=1)

                ###

                scores_hands = torch.max(classification_hands[index, :, :].unsqueeze(0), dim=2, keepdim=True)[0]
                scores_over_thresh_hands = (scores_hands.squeeze() > 0.05) #checks scores for each image

                if scores_over_thresh_hands.sum() == 0:
                    all_nms_scores.append(torch.zeros(1, 1))
                    all_nms_classes.append(torch.zeros(1, 1))
                    all_transformed_anchors.append(torch.zeros(1, 1))
                    continue

                image_classification_hands = classification_hands[index, scores_over_thresh_hands, :]
                image_transformed_anchors_hands = transformed_anchors[index, scores_over_thresh_hands, :]
                scores_hands = scores_hands.squeeze()[scores_over_thresh_hands]

                anchors_nms_idx_hands = nms(image_transformed_anchors_hands, scores_hands, 0.25)

                nms_scores_hands, nms_class_hands = image_classification_hands[anchors_nms_idx_hands, :].max(dim=1)

                all_nms_scores.append(torch.cat((nms_scores_hands, nms_scores_tools), 0))
                all_nms_classes.append(torch.cat((nms_class_hands, nms_class_tools), 0))
                all_transformed_anchors.append(torch.cat((image_transformed_anchors_hands[anchors_nms_idx_hands, :], image_transformed_anchors_tools[anchors_nms_idx_tools, :]), 0))

            return [all_nms_scores, all_nms_classes, all_transformed_anchors, action_logits]




def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
