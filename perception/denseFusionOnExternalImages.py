import argparse
import copy
import os
import sys

import numpy as np
import numpy.ma as ma
import scipy.io as scio
import torch
import torchvision.transforms as transforms
from PIL import Image
from mmdOnExternalImages import mmdInferenceOnExternalImages
from pycocotools.mask import decode
from torch.autograd import Variable
from tqdm import tqdm

sys.path.append(os.path.join(sys.path[0], "DenseFusion"))
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix

"""This piece of code is adapted from DenseFusion's eval_ycb.py to robot perception.
The major modification is that, the data input to DenseFusion model changes from XXX/results_PoseCNN_RSS2018/XXX.mat
	to the output of mmdetection. """

parser = argparse.ArgumentParser()
parser.add_argument('--ur_perception_ros_path', type=str, help="inside ros package, package/data.")
parser.add_argument('--perception_repos_path', type=str, help="under which there is DenseFusion")

parser.add_argument('--external_inputs_path', type=str, default='external_inputs', help='external images root dir')
parser.add_argument('--mmd_output_result_path', type=str, default='external_outputs/mmd_result',
                    help='mmdetection external images root dir')
parser.add_argument('--mmd_config', type=str, default='mmdetection/configs/ycb_mask_rcnn_r50_fpn_1x.py',
                    help='mmdetection config file')
parser.add_argument('--mmd_checkpoint', type=str, default='mmdetection_work_dirs/ycb_mask_rcnn_r50_fpn_1x/latest.pth',
                    help='trained mmdetection checkpoint')

parser.add_argument('--model', type=str,
                    default='DenseFusion/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth',
                    help='resume PoseNet model')
parser.add_argument('--refine_model', type=str,
                    default='DenseFusion/trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth',
                    help='resume PoseRefineNet model')
config = parser.parse_args()

# create absolute paths out of these relative path. 
external_inputs_path = os.path.join(config.ur_perception_ros_path, config.external_inputs_path)

external_input_images_path = os.path.join(external_inputs_path, "images")
mmd_output_result_path = os.path.join(config.ur_perception_ros_path, config.mmd_output_result_path)
mmd_config = os.path.join(config.perception_repos_path, config.mmd_config)
mmd_checkpoint = os.path.join(config.perception_repos_path, config.mmd_checkpoint)

mmd_results = mmdInferenceOnExternalImages(external_input_images_path, mmd_output_result_path, mmd_config, mmd_checkpoint)

# NOTE : assume images are (480 [height], 640 [width])
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
# TODO : please specify camera intrinsics!
cam_cx = 317.017
cam_cy = 241.722
cam_fx = 615.747
cam_fy = 616.041
cam_scale = 10000.0
num_obj = 21
img_width = 480
img_length = 640
num_points = 1000
num_points_mesh = 500
iteration = 2
bs = 1

# NOTE : specify dense fusion toolbox and outputs.
# TODO : put it in args.
df_output_dir = os.path.join(config.ur_perception_ros_path, 'external_outputs/df_outputs')
result_wo_refine_dir = os.path.join(df_output_dir, 'df_wo_refine_result')
result_refine_dir = os.path.join(df_output_dir, 'df_iterative_result')

if not os.path.exists(result_wo_refine_dir):
	os.makedirs(result_wo_refine_dir)
if not os.path.exists(result_refine_dir):
	os.makedirs(result_refine_dir)

model_checkpoint_path = os.path.join(config.perception_repos_path, config.model)
estimator = PoseNet(num_points=num_points, num_obj=num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(model_checkpoint_path))
estimator.eval()

refiner_checkpoint_path = os.path.join(config.perception_repos_path, config.refine_model)
refiner = PoseRefineNet(num_points=num_points, num_obj=num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(refiner_checkpoint_path))
refiner.eval()

images = os.listdir(os.path.join(external_inputs_path, "images"))


def adjustBboxFormat(bb):
	"""return rmin, rmax, cmin, cmax"""
	left_top = (bb[0], bb[1])
	right_bottom = (bb[2], bb[3])
	# rmin = int(left_top[0]) + 1
	# rmax = int(right_bottom[0]) - 1
	# cmin = int(left_top[1]) + 1
	# cmax = int(right_bottom[1]) - 1
	cmin = int(left_top[0]) + 1
	cmax = int(right_bottom[0]) - 1
	rmin = int(left_top[1]) + 1
	rmax = int(right_bottom[1]) - 1
	r_b = rmax - rmin
	for tt in range(len(border_list)):
		if r_b > border_list[tt] and r_b < border_list[tt + 1]:
			r_b = border_list[tt + 1]
			break
	c_b = cmax - cmin
	for tt in range(len(border_list)):
		if c_b > border_list[tt] and c_b < border_list[tt + 1]:
			c_b = border_list[tt + 1]
			break
	center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
	rmin = center[0] - int(r_b / 2)
	rmax = center[0] + int(r_b / 2)
	cmin = center[1] - int(c_b / 2)
	cmax = center[1] + int(c_b / 2)
	if rmin < 0:
		delt = -rmin
		rmin = 0
		rmax += delt
	if cmin < 0:
		delt = -cmin
		cmin = 0
		cmax += delt
	if rmax > img_width:
		delt = rmax - img_width
		rmax = img_width
		rmin -= delt
	if cmax > img_length:
		delt = cmax - img_length
		cmax = img_length
		cmin -= delt
	return rmin, rmax, cmin, cmax


with tqdm(total=len(images)) as pbar:
	for now, image_name in enumerate(images):
		img = Image.open(os.path.join(external_inputs_path, "images", image_name))
		depth = np.array(Image.open(os.path.join(external_inputs_path, "depths", image_name)))
		
		"""Here we need to translate the output of mmdetection to the form consistent with PoseCNN result,
			therefore we gotta give a description of them.
		In original DenseFusion:
		For each image, several objects can be detected, thus we would have a list of items.
		Each item consists of class id, segmentation mask, and bounding box.
		Segmentation mask can be represented compactly in an image-like numpy array, where is pixel value is the class index.
		Mask min is 0, mask max is 21, to represent 21 classes and null class.
		
		The output of mmdetection:
		The output mmd_results in a dict, key is image name, value is a tuple (bboxes, labels).
		bboxes is a list of length 21, each element is a numpy array (N, 5), where the first 4 of 5 is bounding box,
			the last one may be confidence score. While N means there may be several objects within identical category.
		labels is a list of length 21, each element is a list of instance masks, which is a dict {'size': XXX, 'counts': XXX}.
        mmdetection's bbox is represented as
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3]).
		"""
		# segmentation label.
		cls_bboxes, cls_segms = mmd_results[image_name]
		
		my_result_wo_refine = []
		my_result = []
		result_meta = {"name": image_name, "classIds": []}
		
		# enumerate over classes.
		for cls_id in range(len(cls_bboxes)):
			itemid = cls_id + 1
			bboxes = cls_bboxes[cls_id]
			segms = cls_segms[cls_id]
			# enumerate over instances within this class.
			for inst in range(bboxes.shape[0]):
				try:
					bbox = bboxes[inst, :]
					segm = segms[inst]
					
					rmin, rmax, cmin, cmax = adjustBboxFormat(bbox)
					
					mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
					mask_label = decode(segm).astype(np.bool)
					mask = mask_label * mask_depth

					choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
					print("len choose is ", len(choose))
					if len(choose) > num_points:
						c_mask = np.zeros(len(choose), dtype=int)
						c_mask[:num_points] = 1
						np.random.shuffle(c_mask)
						choose = choose[c_mask.nonzero()]
					else:
						choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')
					
					depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
					xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
					ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
					choose = np.array([choose])
					
					pt2 = depth_masked / cam_scale
					pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
					pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
					cloud = np.concatenate((pt0, pt1, pt2), axis=1)
					
					img_masked = np.array(img)[:, :, :3]
					img_masked = np.transpose(img_masked, (2, 0, 1))
					img_masked = img_masked[:, rmin:rmax, cmin:cmax]
					
					cloud = torch.from_numpy(cloud.astype(np.float32))
					choose = torch.LongTensor(choose.astype(np.int32))
					img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
					index = torch.LongTensor([cls_id])
					
					cloud = Variable(cloud).cuda()
					choose = Variable(choose).cuda()
					img_masked = Variable(img_masked).cuda()
					index = Variable(index).cuda()
					
					cloud = cloud.view(1, num_points, 3)
					img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])
					
					pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
					pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
					
					pred_c = pred_c.view(bs, num_points)
					how_max, which_max = torch.max(pred_c, 1)
					pred_t = pred_t.view(bs * num_points, 1, 3)
					points = cloud.view(bs * num_points, 1, 3)
					
					my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
					my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
					my_pred = np.append(my_r, my_t)
					my_result_wo_refine.append([cls_id] + my_pred.tolist())
					
					for ite in range(0, iteration):
						T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points,
						                                                                                 1).contiguous().view(
							1, num_points, 3)
						my_mat = quaternion_matrix(my_r)
						R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
						my_mat[0:3, 3] = my_t
						
						new_cloud = torch.bmm((cloud - T), R).contiguous()
						pred_r, pred_t = refiner(new_cloud, emb, index)
						pred_r = pred_r.view(1, 1, -1)
						pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
						my_r_2 = pred_r.view(-1).cpu().data.numpy()
						my_t_2 = pred_t.view(-1).cpu().data.numpy()
						my_mat_2 = quaternion_matrix(my_r_2)
						
						my_mat_2[0:3, 3] = my_t_2
						
						my_mat_final = np.dot(my_mat, my_mat_2)
						my_r_final = copy.deepcopy(my_mat_final)
						my_r_final[0:3, 3] = 0
						my_r_final = quaternion_from_matrix(my_r_final, True)
						my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])
						
						my_pred = np.append(my_r_final, my_t_final)
						my_r = my_r_final
						my_t = my_t_final
					
					my_result.append([cls_id] + my_pred.tolist())
				except ZeroDivisionError:
					print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))
					my_result_wo_refine.append([cls_id] + [0.0 for i in range(7)])
					my_result.append([cls_id] + [0.0 for i in range(7)])
				
		"the saved pose result. 0th position gives class id."
		scio.savemat('{0}/{1}.mat'.format(result_wo_refine_dir, image_name),
		             {'poses': my_result_wo_refine})
		scio.savemat('{0}/{1}.mat'.format(result_refine_dir, image_name), {'poses': my_result})
		print("Finish No.{0} keyframe".format(now))
	pbar.update(1)
