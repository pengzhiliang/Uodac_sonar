from .coco import CocoDataset
from .registry import DATASETS
import numpy as np

@DATASETS.register_module
class SonarDataset(CocoDataset):
    CLASSES = ('target',)

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 filter_type_prefix=None,
                 ):
        self.use_nagetive = False
        self.filter_type_prefix = filter_type_prefix
        self.negative_filter_type = ('gx', 'fl')
        self.bad_negative=['ss-355_0x0.jpg', 'ss-121_0x0.jpg', 'ss-121_1x0.jpg', 'ss-122_1x0.jpg',
                           'ss-123_0x0.jpg', 'ss-180_1x0.jpg', 'ss-282_1x0.jpg', 'ss-284_0x0.jpg',
                           'ss-313_0x0.jpg', 'ss-339_0x0.jpg', 'ss-353_1x0.jpg', 'ss-355_0x0.jpg']

        for p in pipeline:
            if p['type'] == 'Grafting_SideScan':
                if p.get('use_side', False):
                    self.use_nagetive = True
        super(SonarDataset, self).__init__(ann_file,
                                           pipeline,
                                           data_root,
                                           img_prefix,
                                           seg_prefix,
                                           proposal_file,
                                           test_mode,
                                           filter_empty_gt)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = []
        for a in self.coco.anns.values():
            if self.use_nagetive:  # use negative img
                ids_with_ann.append(a['image_id'])
            else:  # Not use negative image
                if a.get('negative', 0) == 0:
                    ids_with_ann.append(a['image_id'])
        ids_with_ann = set(ids_with_ann)

        # ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if self.filter_type_prefix is not None:
                if img_info['file_name'][:2] in self.filter_type_prefix:
                    continue
            if self.use_nagetive and (self.negative_filter_type is not None) and (img_info.get('negative', 0) == 1):
                if (img_info['file_name'][:2] in self.negative_filter_type):
                    continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_bboxes_weight = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            
#             gt_bboxes_weight.append(ann.get('weight', 1.))
            
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')
#         print(gt_bboxes_weight)
        ann = dict(
            bboxes=gt_bboxes,
#             gt_bboxes_weight=gt_bboxes_weight,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann