import argparse
import os

from mmdet.apis import init_detector, inference_detector, show_result

parser = argparse.ArgumentParser()
parser.add_argument('--external_images_path', type=str, default='external_images', help='external images root dir')
parser.add_argument('--output_result_path', type=str, default='external_outputs', help='external images root dir')
config = parser.parse_args()

config_file = 'mmdetection/configs/ycb_mask_rcnn_r50_fpn_1x.py'
checkpoint_file = 'mmdetection_work_dirs/ycb_mask_rcnn_r50_fpn_1x/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
base_path = os.getcwd()
external_images_path = os.path.join(base_path, config.external_images_path)
external_outputs_path = os.path.join(base_path, config.output_result_path)
images = os.listdir(external_images_path)

from tqdm import tqdm

with tqdm(total=len(images)) as pbar:
	for image_name in images:
		image_path = os.path.join(external_images_path, image_name)
		result = inference_detector(model, image_path)
		output_path = os.path.join(external_outputs_path, image_name)
		# visualize the results in a new window
		# show_result(image_path, result, model.CLASSES)
		# or save the visualization results to image files
		show_result(image_path, result, model.CLASSES, show=False, out_file=output_path)
		pbar.update(1)
