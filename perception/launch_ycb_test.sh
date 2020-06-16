#!/bin/bash 

python tools/test.py configs/ycb_mask_rcnn_r50_fpn_1x.py ./mmdetection_work_dirs/ycb_mask_rcnn_r50_fpn_1x/latest.pth --eval bbox segm --show
