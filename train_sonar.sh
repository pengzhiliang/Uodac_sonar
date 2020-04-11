#!/usr/bin/env bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0 #,1,2,3,4,5,6,7
GPUS=1 # 8

    
DIR=work_dirs/cascade_rcnn_dconv_c3-c5_r101_uncertain_MS_dufpn_1x
CONFIG=configs/cascade_rcnn_dconv_c3-c5_r101_uncertain_MS_dufpn_1x.py
./tools/dist_train.sh ${CONFIG} ${GPUS} --work_dir ${DIR} --gpus ${GPUS} --autoscale-lr --validate
