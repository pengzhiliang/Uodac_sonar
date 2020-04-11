#!/usr/bin/env bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0 #,1,2,3,4,5,6,7
GPUS=1 #8

    
DIR=work_dirs/cascade_rcnn_dconv_c3-c5_r101_uncertain_MS_dufpn_1x
CONFIG=configs/cascade_rcnn_dconv_c3-c5_r101_uncertain_MS_dufpn_1x.py
./tools/dist_test.sh ${CONFIG} pretrain_model/cascade_rcnn_dconv_c3-c5_r101_uncertain_MS_dufpn_1x_sonar_final.pth ${GPUS} --out ${DIR}/bbox_predict.pkl --eval bbox 
# ./tools/dist_test.sh ${CONFIG} ${DIR}/latest.pth ${GPUS} --out ${DIR}/bbox_predict.pkl --eval bbox 

# python ./tools/vis_det.py
# python ./tools/turn_pkl_to_csv.py ${DIR}/bbox_predict.pkl --json_path data/annotation/a-test.json