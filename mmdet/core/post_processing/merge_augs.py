import numpy as np
import torch

from mmdet.ops import nms
from ..bbox import bbox_mapping_back


def merge_aug_proposals(aug_proposals, img_metas, rpn_test_cfg):
    """Merge augmented proposals (multiscale, flip, etc.)

    Args:
        aug_proposals (list[Tensor]): proposals from different testing
            schemes, shape (n, 5). Note that they are not rescaled to the
            original image size.

        img_metas (list[dict]): list of image info dict where each dict has:
            'img_shape', 'scale_factor', 'flip', and my also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `mmdet/datasets/pipelines/formatting.py:Collect`.

        rpn_test_cfg (dict): rpn test config.

    Returns:
        Tensor: shape (n, 4), proposals corresponding to original image scale.
    """
    recovered_proposals = []
    for proposals, img_info in zip(aug_proposals, img_metas):
        img_shape = img_info['img_shape']
        scale_factor = img_info['scale_factor']
        flip = img_info['flip']
        flip_direction = img_info.get('flip_direction', 'horizontal')
        _proposals = proposals.clone()
        _proposals[:, :4] = bbox_mapping_back(_proposals[:, :4], img_shape,
                                              scale_factor, flip, flip_direction)
#         recovered_proposals.append(_proposals)
        w = _proposals[:, 2] - _proposals[:, 0]
        h = _proposals[:, 3] - _proposals[:, 1]
        scale = (w * h).sqrt()
#         if scale_factor <= 1.0: # detetion large object on small scale
#             inds = (w * h).sqrt() >= 128.
#         else: # detection small object on large scale 
#             inds = (w * h).sqrt() <= 256.
#         recovered_proposals.append(_proposals[inds])
        if scale_factor < 1.0 - 0.05:
            inds = scale >= 64
            recovered_proposals.append(_proposals[inds])
        elif scale_factor > 1.0 + 0.05:
            inds = scale <= 400
            recovered_proposals.append(_proposals[inds])
        else:
            recovered_proposals.append(_proposals)
        
    aug_proposals = torch.cat(recovered_proposals, dim=0)
    merged_proposals, _ = nms(aug_proposals, rpn_test_cfg.nms_thr)
    scores = merged_proposals[:, 4]
    _, order = scores.sort(0, descending=True)
    num = min(rpn_test_cfg.max_num, merged_proposals.shape[0])
    order = order[:num]
    merged_proposals = merged_proposals[order, :]
    return merged_proposals


def merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg):
    """Merge augmented detection bboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 4*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
        img_shapes (list[Tensor]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        flip_direction = img_info[0].get('flip_direction', 'horizontal')
        bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip, flip_direction)
        recovered_bboxes.append(bboxes)
    bboxes = torch.stack(recovered_bboxes).mean(dim=0)
    if aug_scores is None:
        return bboxes
    else:
        scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores

def merge_aug_bboxes_variance(aug_bboxes, aug_scores, aug_variance, img_metas, rcnn_test_cfg):
    """Merge augmented detection bboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 4*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
        img_shapes (list[Tensor]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_bboxes = []
#     for bboxes, img_info in zip(aug_bboxes, img_metas):
    for i, (bboxes, img_info) in enumerate(zip(aug_bboxes, img_metas)):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        flip_direction = img_info[0].get('flip_direction', 'horizontal')
        bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip, flip_direction)
#         recovered_bboxes.append(bboxes)
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        scale = (w * h).sqrt()
        if scale_factor < 1.0 - 0.05:
            inds = scale <= 64. 
            bboxes[inds] = 0.
            aug_scores[i][inds] = 0.
            aug_variance[i][inds] = 0.
        elif scale_factor > 1.0 + 0.05:
            inds = scale >= 400.
            bboxes[inds] = 0.
            aug_scores[i][inds] = 0.
            aug_variance[i][inds] = 0.
        recovered_bboxes.append(bboxes)


#     bboxes = torch.stack(recovered_bboxes).mean(dim=0) # mean
#     scores = torch.stack(aug_scores).mean(dim=0) # mean
#     variance = torch.stack(aug_variance).mean(dim=0) # mean

    bboxes = torch.stack(recovered_bboxes)#.sum(dim=0) # mean
    scores = torch.stack(aug_scores)#.sum(dim=0) # mean
    variance = torch.stack(aug_variance)#.sum(dim=0) # mean
    mean_factor = (bboxes.sum(-1) > 0).sum(0)[:, None].float()
    bboxes = bboxes.sum(0) / mean_factor
    scores = scores.sum(0) / mean_factor
    variance = variance.sum(0) / mean_factor
    
    # var_weight = torch.stack(aug_variance).sum(-1)
    #
    # var_weight = 1 / torch.stack(aug_variance)  # variance weighted
    # var_weight /= var_weight.sum(0)  # variance weighted
    # bboxes = (torch.stack(recovered_bboxes) * var_weight).sum(0) # variance weighted
    # variance = (torch.stack(aug_variance) * var_weight).sum(0) # variance weighted
    #
    # var_weight = (1 / torch.stack(aug_variance)).sum(-1) # variance weighted
    # var_weight /= var_weight.sum(0) # variance weighted
    # scores = (torch.stack(aug_scores) * var_weight[..., None]).sum(0) # variance weighted

    # var_argmin = torch.stack(aug_variance).sum(-1).argmin(0)[:, None].float()
    # var_argmin = torch.stack([1 - var_argmin, var_argmin])
    # bboxes = (torch.stack(recovered_bboxes) * var_argmin).sum(0)
    # scores = (torch.stack(aug_scores) * var_argmin).sum(0)
    # variance = (torch.stack(aug_variance) * var_argmin).sum(0)
    return bboxes, scores, variance

def merge_aug_scores(aug_scores):
    """Merge augmented bbox scores."""
    if isinstance(aug_scores[0], torch.Tensor):
        return torch.mean(torch.stack(aug_scores), dim=0)
    else:
        return np.mean(aug_scores, axis=0)


def merge_aug_masks(aug_masks, img_metas, rcnn_test_cfg, weights=None):
    """Merge augmented mask prediction.

    Args:
        aug_masks (list[ndarray]): shape (n, #class, h, w)
        img_shapes (list[ndarray]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_masks = [
        mask if not img_info[0]['flip'] else mask[..., ::-1]
        for mask, img_info in zip(aug_masks, img_metas)
    ]
    if weights is None:
        merged_masks = np.mean(recovered_masks, axis=0)
    else:
        merged_masks = np.average(
            np.array(recovered_masks), axis=0, weights=np.array(weights))
    return merged_masks
