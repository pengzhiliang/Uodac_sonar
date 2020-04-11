from __future__ import division
import argparse
import copy
import os
import glob
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument(
        '--cross-validate',
        action='store_true',
        help='whether to perform cross validate or not')
    parser.add_argument(
        '--clear-output',
        action='store_true',
        help='whether to clear cross validate output weight or not')
    parser.add_argument('--left_parameters', nargs=argparse.REMAINDER, help='all the other parameters')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def merge_from_list(cfg, cfg_list):
    import re
    assert len(cfg_list) % 2 == 0, "it must be a list of pairs"
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        if v.isdigit():
            v = int(v)
        else:
            value = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
            if value.match(v):
                v = float(v)
        sp_key = full_key.split('.')
        if len(sp_key) == 1:
            cfg[sp_key[0]] = v
        elif len(sp_key) == 2:
            cfg[sp_key[0]][sp_key[1]] = v
        elif len(sp_key) == 3:
            cfg[sp_key[0]][sp_key[1]][sp_key[2]] = v
        else:
            cfg[sp_key[0]][sp_key[1]][sp_key[2]][sp_key[3]] = v
    return cfg

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.cross_validate:
        for i in range(4):
            if os.path.exists(cfg.data.train.ann_file.replace('re_', '{}_split_'.format(str(i+1)))):
                continue
            else:
                raise FileNotFoundError(cfg.data.train.ann_file.replace('re_', '{}_split_'.format(str(i+1))))

    if args.left_parameters is not None:
        cfg = merge_from_list(cfg, args.left_parameters)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = (cfg.optimizer['lr'] * cfg.gpus / 8) * cfg.data.imgs_per_gpu / 2

    # init distributed env first, since logger depends on the dist info.

    if args.cross_validate:
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)
        original_ann = cfg.data.train.ann_file
        original_output = cfg.work_dir
        for i in range(0, 4):
            now_ann = original_ann.replace('re_', '{}_split_'.format(str(i+1)))
            now_output = os.path.join(original_output, '{}_split'.format(str(i+1)))

            cfg.data.train.ann_file = now_ann
            cfg.work_dir = now_output

            # create work_dir
            mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
            # init the logger before other steps
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
            logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

            # init the meta dict to record some important information such as
            # environment info and seed, which will be logged
            meta = dict()
            # log env info
            env_info_dict = collect_env()
            env_info = '\n'.join([('{}: {}'.format(k, v))
                                  for k, v in env_info_dict.items()])
            dash_line = '-' * 60 + '\n'
            logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                        dash_line)
            meta['env_info'] = env_info

            # log some basic info
            logger.info('Distributed training: {}'.format(distributed))
            logger.info('Config:\n{}'.format(cfg.text))

            # set random seeds
            if args.seed is not None:
                logger.info('Set random seed to {}, deterministic: {}'.format(
                    args.seed, args.deterministic))
                set_random_seed(args.seed, deterministic=args.deterministic)
            cfg.seed = args.seed
            meta['seed'] = args.seed

            model = build_detector(
                cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

            datasets = [build_dataset(cfg.data.train)]
            if len(cfg.workflow) == 2:
                val_dataset = copy.deepcopy(cfg.data.val)
                val_dataset.pipeline = cfg.data.train.pipeline
                datasets.append(build_dataset(val_dataset))
            if cfg.checkpoint_config is not None:
                # save mmdet version, config file content and class names in
                # checkpoints as meta data
                cfg.checkpoint_config.meta = dict(
                    mmdet_version=__version__,
                    config=cfg.text,
                    CLASSES=datasets[0].CLASSES)
            # add an attribute for visualization convenience
            model.CLASSES = datasets[0].CLASSES
            train_detector(
                model,
                datasets,
                cfg,
                distributed=distributed,
                validate=args.validate,
                timestamp=timestamp,
                meta=meta)

            if args.clear_output:
                max_epochs = cfg.total_epochs
                files = glob.glob(now_output+'/epoch_*.pth')
                for file in files:
                    e = file.split('_')[-1]
                    e = int(e.split('.')[0])
                    if e < max_epochs:
                        os.remove(file)


    else:
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)
        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([('{}: {}'.format(k, v))
                              for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info

        # log some basic info
        logger.info('Distributed training: {}'.format(distributed))
        logger.info('Config:\n{}'.format(cfg.text))

        # set random seeds
        if args.seed is not None:
            logger.info('Set random seed to {}, deterministic: {}'.format(
                args.seed, args.deterministic))
            set_random_seed(args.seed, deterministic=args.deterministic)
        cfg.seed = args.seed
        meta['seed'] = args.seed

        model = build_detector(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__,
                config=cfg.text,
                CLASSES=datasets[0].CLASSES)
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=args.validate,
            timestamp=timestamp,
            meta=meta)


if __name__ == '__main__':
    main()
