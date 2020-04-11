
import sys
import os
import shutil
import numpy as np
import json
import xml.etree.ElementTree as ET
import mmcv
from tqdm import tqdm
from glob import glob

#
START_IMAGE_ID = 1
WITHOUT_IMAGE_INFO = True



def get(root, name):
    vars = root.findall(name)
    return vars

def convert(img_list, image_dir, json_file):
    json_dict = {"images": [],
                 "type": "instances",
                 "annotations": [],
                 "categories": []}

    image_id = START_IMAGE_ID - 1
    for line in tqdm(img_list):
        filename = line.split('.')[0]
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
        json_dict['images'].append(image)

    cat = {'supercategory': 'none', 'id': 1, 'name': 'target'}
    json_dict['categories'].append(cat)
    print(json_dict['categories'])
    print('end image id: {} total images: {}'.format(image_id, len(json_dict['images'])))
    # 导出到json
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == '__main__':

    root_path = 'YOUR PATH /a-test-image'
    img_dir = os.path.join(root_path, 'image', 'combine')

    img_list = os.listdir(os.path.join(root_path, 'image', 'combine'))

    json_file = 'YOUR PATH /a-test.json'

    convert(img_list, img_dir, json_file)
