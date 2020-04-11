from mmdet.apis import init_detector, inference_detector, show_result_pyplot, project_dir
import mmcv, os, csv
import os.path as osp
import json, pickle
import numpy as np
import matplotlib.pyplot as plt
import pylab
import cv2

from pycocotools.coco import COCO
from mmcv.visualization.color import color_val
from tqdm import tqdm
import pdb


# pdb.set_trace()
def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles

def coco_anns_to_label_box(anns):
    # anns : [ann1, ann2]
    # ann format
    # {'area': 9558,
    #   'bbox': [331, 102, 162, 59],
    #   'category_id': 13,
    #   'id': 14940,
    #   'ignore': 0,
    #   'image_id': 9931,
    #   'iscrowd': 0,
    #   'segmentation': [[331, 102, 331, 161, 493, 161, 493, 102]]}
    instance_num = len(anns)
    boxes = np.zeros((instance_num, 4))
    labels = np.zeros(instance_num)
    for i, ann in enumerate(anns):
        box = ann['bbox']
        boxes[i] = [box[0], box[1], box[0] + box[2] - 1, box[1] + box[3] - 1]
        labels[i] = ann['category_id']
    return labels, boxes


def get_class_names(cats, start_class=1):
    # {1: {'id': 1, 'name': 'aeroplane', 'supercategory': 'none'},
    #  2: {'id': 2, 'name': 'bicycle', 'supercategory': 'none'},}
    class_names = []
    class_num = len(cats.keys())
    for i in range(start_class, class_num + start_class):
        class_names.append(cats[i]['name'])
    return class_names


def show_bbox(img, labels, bboxes, class_names=None, show_threshold=0.1, bbox_color='green', text_color='green', no_name=False):
    # labels >= 1
    thickness = 2
    font_scale = 1
    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    if show_threshold > 0 and bboxes.shape[1] == 5:
        labels = labels[bboxes[:, -1] > show_threshold]
        bboxes = bboxes[bboxes[:, -1] > show_threshold]

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        if no_name:
            if len(bbox) > 4:
                if bbox_int[1] > 20:
                    txt_loac = (bbox_int[0], bbox_int[1])
                else:
                    txt_loac = (bbox_int[0], bbox_int[3])
                cv2.putText(img, '{:.02f}'.format(bbox[-1]), txt_loac, cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        else:
            assert class_names is not None
            label_text = class_names[int(label.item())] if class_names is not None else 'cls {}'.format(label)
            if len(bbox) > 4:
                label_text += '|{:.02f}'.format(bbox[-1])
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    return img


if __name__ == '__main__':

    bbox_predict_file = osp.join(project_dir, 'work_dirs/cascade_rcnn_dconv_c3-c5_r101_uncertain_MS_dufpn_1x/bbox_predict.pkl')

    img_dir = osp.join(project_dir, 'data/a-test-image')
    test_dataset_path = osp.join(project_dir, 'data/annotation/a-test.json')
    save_dir = osp.join(project_dir, 'data/a_image_det_vis')
    
    # save_dir = None
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    with open(test_dataset_path) as f:
        test_json = json.load(f)
    with open(bbox_predict_file, 'rb') as f:
        bbox_predict = pickle.load(f)
    
    det_color = 'blue'
    show_thr = 0.3
    
    assert len(bbox_predict) == len(test_json['images'])
    for img_info, bbox in tqdm(zip(test_json['images'], bbox_predict)):

        img_path = os.path.join(img_dir, img_info['file_name'])
        I = mmcv.imread(img_path)
        
        if img_info['file_name'][0] == 'a':
            show_thr = 0.1
        else:
            show_thr = 0.3

        det_result = bbox[0]
        I_add_det = show_bbox(I, np.ones(det_result.shape[0]), det_result,
                                 show_threshold=show_thr, bbox_color=det_color, text_color=det_color, no_name=True)

        cv2.imwrite(os.path.join(save_dir, img_info['file_name']), I_add_det)
