import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms)
from mmdet.ops import nms
from ..registry import HEADS
from .rpn_head import RPNHead


@HEADS.register_module
class RPNHeadWeed(RPNHead):

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    # def loss(self,
    #          cls_scores,
    #          bbox_preds,
    #          gt_bboxes,
    #          gt_labels,
    #          img_metas,
    #          cfg,
    #          gt_bboxes_ignore=None):
    #     featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    #     assert len(featmap_sizes) == len(self.anchor_generators)
    #
    #     device = cls_scores[0].device
    #
    #     anchor_list, valid_flag_list = self.get_anchors(
    #         featmap_sizes, img_metas, device=device)
    #     label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
    #     cls_reg_targets = anchor_target(
    #         anchor_list,
    #         valid_flag_list,
    #         gt_bboxes,
    #         img_metas,
    #         self.target_means,
    #         self.target_stds,
    #         cfg,
    #         gt_bboxes_ignore_list=gt_bboxes_ignore,
    #         gt_labels_list=gt_labels,
    #         label_channels=label_channels,
    #         sampling=self.sampling)
    #     if cls_reg_targets is None:
    #         return None
    #     (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
    #      num_total_pos, num_total_neg) = cls_reg_targets
    #     num_total_samples = (
    #         num_total_pos + num_total_neg if self.sampling else num_total_pos)
    #     losses_cls, losses_bbox = multi_apply(
    #         self.loss_single,
    #         cls_scores,
    #         bbox_preds,
    #         labels_list,
    #         label_weights_list,
    #         bbox_targets_list,
    #         bbox_weights_list,
    #         num_total_samples=num_total_samples,
    #         cfg=cfg)
    #     return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox
