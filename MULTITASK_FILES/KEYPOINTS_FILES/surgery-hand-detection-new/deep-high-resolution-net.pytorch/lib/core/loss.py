# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
    #     heatmaps = []
    #     valid = []

    #     keypoint_side_len = output.shape[2]
    #     for instances_per_image in instances:
    #         if len(instances_per_image) == 0:
    #             continue
    #         keypoints = instances_per_image.gt_keypoints
    #         heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
    #             instances_per_image.proposal_boxes.tensor, keypoint_side_len
    #         )
    #         heatmaps.append(heatmaps_per_image.view(-1))
    #         valid.append(valid_per_image.view(-1))

    # if len(heatmaps):
    #     keypoint_targets = cat(heatmaps, dim=0)
    #     valid = cat(valid, dim=0).to(dtype=torch.uint8)
    #     valid = torch.nonzero(valid).squeeze(1)

    # # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # # accept empty tensors, so handle it separately
    # if len(heatmaps) == 0 or valid.numel() == 0:
    #     global _TOTAL_SKIPPED
    #     _TOTAL_SKIPPED += 1
    #     storage = get_event_storage()
    #     storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
    #     return pred_keypoint_logits.sum() * 0

    # N, K, H, W = pred_keypoint_logits.shape
    # pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)

    # keypoint_loss = F.cross_entropy(
    #     pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    # )

    # # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    # if normalizer is None:
    #     normalizer = valid.numel()
    # keypoint_loss /= normalizer

    # return keypoint_loss

        # batch_size, num_joints, h, w = output.shape
        # heatmaps_pred = output.reshape((batch_size * num_joints, h * w))
        # heatmaps_gt = target.reshape((batch_size * num_joints, h * w))

        # loss = F.cross_entropy(
        #     pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
        # )

        # for idx in range(num_joints):
        #     heatmap_pred = heatmaps_pred[idx].squeeze()
        #     heatmap_gt = heatmaps_gt[idx].squeeze()
        #     if self.use_target_weight:
        #         #num_visible = torch.sum(target_weight[:, idx])
        #         #if num_visible == 0:
        #             #continue
        #         num_visible = 1
        #         loss += self.criterion(
        #             heatmap_pred.mul(target_weight[:, idx]),
        #             heatmap_gt.mul(target_weight[:, idx])
        #             ) / num_visible
        #     else:
        #         loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        # return loss
        # batch_size, num_joints, h, w = output.shape
        # heatmaps_pred = output.reshape((batch_size * num_joints, h, w))
        # heatmaps_gt = target.reshape((batch_size * num_joints, h, w))
        # # print(heatmaps_gt[0, 1])
        # loss = 0

        # # target weight 16 x 21 x 1
        # valid = torch.nonzero(target_weight.squeeze(-1))
        # # idxs = valid[:, 0] * 21 + valid[:, 1]

        # x = heatmaps_pred[idxs]
        # y = heatmaps_gt[idxs].float()
        # # loss = torch.mean(torch.sum(y * x, 1))

        # # loss = F.cross_entropy(
        # #     x, y, reduction="sum"
        # # )

        # loss = self.criterion(x, y)

        # normalizer = idxs.numel()
        # loss /= normalizer

        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        num_visible = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                # num_visible += torch.sum(target_weight[:, idx])
                loss += self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                    ) 
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss * 100 / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
