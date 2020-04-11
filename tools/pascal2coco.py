# -*- coding=utf-8 -*-
# !/usr/bin/python

import sys
import os
import shutil
import numpy as np
import json
import xml.etree.ElementTree as ET
import mmcv
from tqdm import tqdm
from glob import glob

from mmdet.apis import project_dir
START_BOUNDING_BOX_ID = 1
#
START_IMAGE_ID = 1
USE_NAME_AS_IMAGE_ID = False
WITHOUT_IMAGE_INFO = True

# construct box for negative image
NEGATIVE_BOX = True

PRE_DEFINE_CATEGORIES = {}



def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


# 得到图片唯一标识号
def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.' % (filename))


def convert(xml_list, xml_dir, image_dir, json_file):
    '''
    :param xml_list: 需要转换的XML文件列表
    :param xml_dir: XML的存储文件夹
    :param json_file: 导出json文件的路径
    :return: None
    '''
    list_fp = xml_list
    # 标注基本结构
    json_dict = {"images": [],
                 "type": "instances",
                 "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    image_id = START_IMAGE_ID - 1
    for line in tqdm(list_fp):
        line = line.strip()
        # print("buddy~ Processing {}".format(line))
        # 解析XML
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'frame')
        # 取出图片名字
        filename = line.split('.')[0]
        ## The filename must be a number
        if USE_NAME_AS_IMAGE_ID:
            image_id = get_filename_as_int(filename)  # 图片ID
        else:
            image_id += 1
        try:
            filename += '.jpg'
            img_path = os.path.join(image_dir, filename)
            img = mmcv.imread(img_path)
        except FileNotFoundError:
            filename = line.split('.')[0] + '.JPG'
            img_path = os.path.join(image_dir, filename)
            img = mmcv.imread(img_path)
        height, width, _ = img.shape
        image = {'file_name': filename,
                 'height': height,
                 'width': width,
                 'id': image_id}

        # 处理每个标注的检测框
        if len(get(root, 'object')) == 0 and NEGATIVE_BOX:
            # negative image
            image['negative'] = 1
            assert filename[:2] in ('ss', 'fl', 'gx')
            xmin, ymin, xmax, ymax = 0, 0, width, height
            annotation = dict()
            annotation['area'] = width * height
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [xmin, ymin, width, height]
            annotation['category_id'] = 1
            annotation['id'] = bnd_id
            annotation['ignore'] = 0
            # flag
            annotation['negative'] = 1
            # 设置分割数据，点的顺序为逆时针方向
            annotation['segmentation'] = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]

            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1
            # continue
        else:
            if NEGATIVE_BOX:
                image['negative'] = 0

        json_dict['images'].append(image)

        for obj in get(root, 'object'):
            # 取出检测框类别名称
            category = get_and_check(obj, 'name', 1).text
            # 更新类别ID字典
            # if category == 'waterweeds':  # exclude in validate
            #     if not 'train' in json_file:
            #         continue
            if category == '目标物':
                category = 'target'
            if category not in categories:
                new_id = len(categories) + 1
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            if xmax - xmin <= 2 or ymax - ymin <= 2:
                continue
            assert (xmax > xmin)
            assert (ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            annotation = dict()
            annotation['area'] = o_width * o_height
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [xmin, ymin, o_width, o_height]
            annotation['category_id'] = category_id
            annotation['id'] = bnd_id
            annotation['ignore'] = 0
            # flag
            if NEGATIVE_BOX:
                annotation['negative'] = 0
            # 设置分割数据，点的顺序为逆时针方向
            annotation['segmentation'] = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]

            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1

    # 写入类别ID字典
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    print(json_dict['categories'])
    print('end image id: {} total images: {}'.format(image_id, len(json_dict['images'])))
    print('eng bnx id: {} total instances: {}'.format(bnd_id, len(json_dict['annotations'])))
    if NEGATIVE_BOX:
        print('negative images : ', sum([a['negative'] for a in json_dict['images']]))
        # for a in json_dict['images']:
        #     if a['negative']:
        #         print(a['file_name'])
    print("statistic the distribution of instance :")
    for name, id in PRE_DEFINE_CATEGORIES.items():
        ins_sum = 0
        for ann in json_dict['annotations']:
            if ann['category_id'] == id:
                ins_sum += 1
        print('class {}: {} instances'.format(name, ins_sum))
    # 导出到json
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == '__main__':
    dataset_root = 'YOUR PATH'
    all_xml_list = dict(train=[], val=[])
    for ind in ['forward','side']: # 'negative',
        root_path =  dataset_root + ind

        xml_list = os.listdir(os.path.join(root_path, 'box'))
        # np.random.shuffle(xml_list)
        xml_list = sorted(xml_list)

        if ind == 'side':
            ss_xml = [i for i in xml_list if i[:2]=='ss']
            print(len(ss_xml))
            split_point = int(len(ss_xml) // 6)
            val_xml_list = ss_xml[:split_point] + ss_xml[len(ss_xml) - split_point:]
            print(len(val_xml_list))
            train_xml_list = [i for i in ss_xml if i not in val_xml_list]
            train_xml_list += [i for i in xml_list if i[:2]!='ss' ]
            all_xml_list['val'].extend(val_xml_list)
            all_xml_list['train'].extend(train_xml_list)
        else:
            split_point = int(len(xml_list) // 8)
            all_xml_list['val'].extend(xml_list[:split_point])
            all_xml_list['val'].extend(xml_list[len(xml_list) - split_point:])
            all_xml_list['train'].extend(xml_list[split_point:len(xml_list) - split_point])

    negative_xml_list = os.listdir(os.path.join(dataset_root, 'negative', 'box'))
    print('negative img: ', len(negative_xml_list))
    all_xml_list['train'].extend(negative_xml_list)

    root_path = dataset_root
    xml_dir = os.path.join(root_path, 'box')
    img_dir = os.path.join(root_path, 'image')

    # xml_list = os.listdir(os.path.join(root_path, 'box'))

    for ind in ('train', 'val'):
        json_file = os.path.join(root_path, 'annotation', 'instances_' + ind + '.json')
        if not os.path.exists(os.path.join(root_path, 'annotation')):
            os.mkdir(os.path.join(root_path, 'annotation'))
        convert(all_xml_list[ind], xml_dir, img_dir, json_file)

    json_file = os.path.join(root_path, 'annotation', 'instances_all.json')
    convert(all_xml_list['train']+all_xml_list['val'], xml_dir, img_dir, json_file)
