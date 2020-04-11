import pickle
import argparse
import json
import csv
import os

import numpy as np
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='turn pkl results to csv to submit')
    parser.add_argument('pkl_path', help='test result file path')
    parser.add_argument('--json_path', help='test json file path', default='data/annotation/a-test.json')
    parser.add_argument('--result_path', help='the path to store csv result', default=None)
    args = parser.parse_args()
    return args


def convert(json_path, pkl_path, result_path=None):
    test_json = json.load(open(json_path))
    test_result = pickle.load(open(pkl_path, 'rb'))  # , encoding='iso-8859-1')
    if len(test_json['categories']) == 1:
        class_name = ('target',)
    else:
        class_name = ('echinus', 'holothurian', 'starfish', 'scallop')
    # pdb.set_trace()
    try:
        assert len(test_json['images']) == len(test_result)
    except AssertionError:
        print('the length of detect pkl result:{}'.format(len(test_result)))
        print('the number image in annotation file:{}'.format(len(test_json)))
        raise AssertionError
    all_result = []

    for img_info, det_result in zip(test_json['images'], test_result):
        for class_id in range(len(class_name)):
            img_name = [class_name[class_id], os.path.basename(img_info['file_name']).replace('jpg', 'xml'), ]
            result = det_result[class_id].tolist()
            for r in result:
                all_result.append(img_name + [r[-1], round(r[0]), round(r[1]), round(r[2]), round(r[3])])
    if result_path is None:
        result_path = pkl_path.replace('pkl', 'csv')

    title = ['name', 'image_id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
    with open(result_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(title)
        writer.writerows(all_result)


if __name__ == '__main__':
    args = parse_args()
    print('Turn json result to csv to submit')
    convert(args.json_path, args.pkl_path, args.result_path)
    result_path = args.result_path
    if args.result_path is None:
        result_path = args.pkl_path.replace('pkl', 'csv')
    print('complete! save at {}'.format(result_path))


