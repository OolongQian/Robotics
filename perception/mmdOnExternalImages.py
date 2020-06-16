import argparse
import os

from mmdet.apis import init_detector, inference_detector, show_result
from tqdm import tqdm


def mmdInferenceOnExternalImages(external_inputs_path, output_result_path, mmd_config_file, mmd_checkpoint):
	# build the model from a config file and a checkpoint file
	model = init_detector(mmd_config_file, mmd_checkpoint, device='cuda:0')
	
	# test a single image and show the results
	base_path = os.getcwd()
	external_inputs_path = os.path.join(base_path, external_inputs_path, "images")
	external_outputs_path = os.path.join(base_path, output_result_path)
	images = os.listdir(external_inputs_path)
	
	results = {}
	
	with tqdm(total=len(images)) as pbar:
		for image_name in images:
			image_path = os.path.join(external_inputs_path, image_name)
			result = inference_detector(model, image_path)
			output_path = os.path.join(external_outputs_path, image_name)
			# visualize the results in a new window
			# show_result(image_path, result, model.CLASSES)
			# or save the visualization results to image files
			show_result(image_path, result, model.CLASSES, show=False, out_file=output_path)
			results[image_name] = result
			pbar.update(1)
	
	return results


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--external_inputs_path', type=str, default='external_inputs', help='external images root dir')
	parser.add_argument('--output_result_path', type=str, default='external_outputs/mmd_result',
	                    help='external images root dir')
	config = parser.parse_args()
	
	config_file = 'mmdetection/configs/ycb_mask_rcnn_r50_fpn_1x.py'
	checkpoint_file = 'mmdetection_work_dirs/ycb_mask_rcnn_r50_fpn_1x/latest.pth'
	
	mmdInferenceOnExternalImages(config.external_inputs_path, config.output_result_path, config_file, checkpoint_file)
