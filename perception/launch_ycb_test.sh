#!/bin/bash 

python mmdetection/tools/test.py mmdetection/configs/ycb_mask_rcnn_r50_fpn_1x.py ./mmdetection_work_dirs/ycb_mask_rcnn_r50_fpn_1x/latest.pth --eval bbox segm --show
