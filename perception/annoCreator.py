#!/usr/bin/env python
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--YCB_dataset_path', type=str, default='datasets/YCB_Video_Dataset', help='YCB dataset root dir')
parser.add_argument('--custom_output_path', type=str, default='datasets/Custom_YCB_Video_Dataset',
                    help='output Custom YCB dataset root dir')

parser.add_argument('--create_trainval', type=str, default='true', help='create annotation for trainval subset')
parser.add_argument('--create_train', type=str, default='true', help='create annotation for train subset')
parser.add_argument('--create_val', type=str, default='true', help='create annotation for val subset')
config = parser.parse_args()

base_path = os.getcwd()
ycb_path = os.path.join(base_path, config.YCB_dataset_path)


# create an index over images. 
# I find /image_sets folder has image index over train, trainval, val. 
#     these can be a kind of image index. 
# YCB index is in the form of 0003/001626, thus I can split them, and 
#     multiply the former part with 10000 to avoid duplication. 

def ycbId2cocoId(folderNo, imageNo):
	"""
	index is the image id in dataset. 
	ycb index -> coco index. 
	"""
	multi = 10000
	return folderNo * multi + imageNo


def createInfo4Datasubset(subset):
	ycb_images_info = []
	image_sets_path = os.path.join(ycb_path, "image_sets")
	with open(os.path.join(image_sets_path, subset)) as f:
		for line in f.readlines():
			line = line.strip('\n')
			folderNoStr, imageNoStr = line.split('/')
			cocoId = ycbId2cocoId(int(folderNoStr), int(imageNoStr))
			ycb_images_info.append({'folder': folderNoStr, 'name': imageNoStr, 'cocoId': cocoId})
	return ycb_images_info


ycb_train_index = "train.txt"
ycb_trainval_index = "trainval.txt"
ycb_val_index = "val.txt"

ycb_trainval_imageInfos = createInfo4Datasubset(ycb_trainval_index)
ycb_train_imageInfos = createInfo4Datasubset(ycb_train_index)
ycb_val_imageInfos = createInfo4Datasubset(ycb_val_index)


# trainval.txt is the union of train.txt and val.txt.

def imageIds2idSet(imageIds):
	ids = []
	for img_info in imageIds:
		ids.append(img_info['cocoId'])
	return ids


# browse images find, rename, and copy them into images folder.
# typical coco image name could be '000000000139'. 

# use the first image info as an example. 

import matplotlib.image as mp


def getImageBasePath(imgInfo):
	config.YCB_dataset_path = "datasets/YCB_Video_Dataset"
	basePath = os.getcwd()
	ycbPath = os.path.join(basePath, config.YCB_dataset_path)
	basePath = os.path.join(ycbPath, "data", imgInfo["folder"], imgInfo["name"])
	return basePath


def getImageColorPath(imgInfo):
	"""get rgb image file name/path"""
	basePath = getImageBasePath(imgInfo)
	folderPath, filename = os.path.split(basePath)
	return os.path.join(folderPath, filename + "-color.png")


def imageId2imageName(imgId):
	return "".join(["0" for _ in range(12 - len(str(imgId)))]) + str(imgId) + ".png"


### this code block is used to play with depth, segLabel, data...

def getBboxPath(imgInfo):
	basePath = getImageBasePath(imgInfo)
	folderPath, filename = os.path.split(basePath)
	return os.path.join(folderPath, filename + "-box.txt")


def ycbBB2cocoBB(bb):
	"""
	# investigate bounding box. 
	# [cmin, rmin, cmax, rmax]
	# coco bounding box is [x, y, width, height]. 
	# (x, y) is the upper-left corner coordinates. 
	# thus the bounding box transformation goes...
	# [cmin, rmin, (cmax-cmin), (rmax-rmin)]"""
	cmin, rmin, cmax, rmax = bb
	return [cmin, rmin, (cmax - cmin), (rmax - rmin)]


def loadBboxes(bboxPath):
	bboxes = []
	with open(bboxPath, 'r') as f:
		for line in f.readlines():
			line = line.strip()
			# note, here we have changed bounding box format. 
			boxCoord = ycbBB2cocoBB([float(coord) for coord in line.split(' ')[1:]])
			catName = line.split(' ')[0]
			bbox = {"name": catName, "bbox": boxCoord}
			bboxes.append(bbox)
	return bboxes


def getSegMaskPath(imgInfo):
	"""get rgb image file name/path"""
	basePath = getImageBasePath(imgInfo)
	folderPath, filename = os.path.split(basePath)
	return os.path.join(folderPath, filename + "-label.png")


### this code block is used to play with meta
from scipy.io import loadmat


def getMetaMatPath(imgInfo):
	"""get rgb image file name/path"""
	basePath = getImageBasePath(imgInfo)
	folderPath, filename = os.path.split(basePath)
	return os.path.join(folderPath, filename + "-meta.mat")


# ### I choose to investigate meta.mat. meta is camera intrinsics.
# #### I find a descent work around. I assume the segmentation mask is used for visualization, thus they do not treat it as significant. 

# this code block is to display the superimposed image and segMask.
def showImagewithSeg(info):
	import matplotlib.pyplot as plt
	imgPath = getImageColorPath(info)
	segPath = getSegMaskPath(info)
	metaPath = getMetaMatPath(info)
	bboxPath = getBboxPath(info)
	img = mp.imread(imgPath)
	seg = mp.imread(segPath)
	meta = loadmat(metaPath)
	bboxes = loadBboxes(bboxPath)
	box = bboxes[0]["bbox"]
	#     print(box)
	# investigate bounding box. 
	# [cmin, rmin, cmax, rmax]
	# coco bounding box is [x, y, width, height]. 
	# (x, y) is the upper-left corner coordinates. 
	# thus the bounding box transformation goes...
	# [cmin, rmin, (cmax-cmin), (rmax-rmin)]
	centers = meta['center']
	xs = [center[0] for center in centers]
	ys = [center[1] for center in centers]
	plt.imshow(img, 'gray', interpolation='none')
	plt.imshow(seg, 'jet', interpolation='none', alpha=0.5)
	plt.scatter(x=xs, y=ys, c='r')  # plt.show()


# for info in ycb_trainval_imageInfos[1053:]:
# 	showImagewithSeg(info)
# 	break

print("the superimposed segmentation looks fairly nice.")

# this code block is used to create RLE segmentation mask.
# this is a rather complicated logic, firstly, we do not factor it into 
#     functions. 

import numpy as np

# for info in ycb_trainval_imageInfos:
# 	imgPath = getImageColorPath(info)
# 	segPath = getSegMaskPath(info)
# 	metaPath = getMetaMatPath(info)
# 	img = mp.imread(imgPath)
# 	seg = mp.imread(segPath)
# 	meta = loadmat(metaPath)
#
# 	#     print(meta.keys())
# 	centers = meta['center']
# 	clsIds = meta['cls_indexes']
#
# 	# iterate through all annotations.
# 	for i in range(len(centers)):
# 		center = centers[i]  # note, center is (x, y).
# 		clsId = clsIds[i]
# 		segVal = seg[int(center[1])][int(center[0])]
#
# 		segMask = np.zeros_like(seg, dtype=np.uint8)
# 		segMask[seg != segVal] = 0
# 		segMask[seg == segVal] = 1
# 		segMask = np.asfortranarray(segMask)
# 		encode(segMask)
# 		assert (decode(encode(segMask)) == segMask).all()
# 		print("we have succeeded to encode segmenetation mask", "to RLE, and ensure their validity. though the data",
# 		      "type is kind of strange, but i am confident that", "it works.")
# 		break
# 	break


# ### then, we care about all classes labels, which can be read in image_sets/classes.txt file. we need to make sure the bbox label, and meta cls_index label are all included in this classes.txt. 

def getAllClassesAndIds():
	base_path = os.getcwd()
	ycb_path = os.path.join(base_path, config.YCB_dataset_path)
	classesPath = os.path.join(ycb_path, "image_sets", "classes.txt")
	
	categories = []
	metaId = 0  # metaId means the class index given in meta.mat's cls_indexes. 
	with open(classesPath, 'r') as f:
		for line in f.readlines():
			line = line.strip()
			#             catId, catName = int(line.split('_')[0]), '_'.join(line.split('_')[1:])
			catName = line
			metaId += 1
			category = {"supercategory": catName,  # "id": catId,
			            #                 "meta_id": metaId,
			            "id": metaId, "name": catName}
			categories.append(category)
	
	return categories


# read all categories. 
cats = getAllClassesAndIds()
allIds = [cat["id"] for cat in cats]
allNames = [cat["name"] for cat in cats]
id2name = {cat["id"]: cat["name"] for cat in cats}
name2id = {cat["name"]: cat["id"] for cat in cats}
# print(cats)

# check the class related information in /data folder are all contained by cats. 
from tqdm import tqdm

with tqdm(total=len(ycb_trainval_imageInfos)) as pbar:
	for info in ycb_trainval_imageInfos:
		metaPath = getMetaMatPath(info)
		bboxPath = getBboxPath(info)
		meta = loadmat(metaPath)
		bboxes = loadBboxes(bboxPath)
		clsIds = meta['cls_indexes'].flatten()
		
		assert len(bboxes) == len(clsIds)
		for i in range(len(bboxes)):
			bbox, clsId = bboxes[i], clsIds[i]
			
			assert clsId in allIds
			assert bbox["name"] in allNames
			
			assert name2id[bbox["name"]] == clsId
		pbar.update(1)
		break

# print("class integrity checking passed. we now have a clear understanding of class name and id.")

# ### class has two attributes, id and name. class.txt file writes all class names, the related class ids are its line number (starting from 1). in /data folder, class appears in -box.txt and -meta.mat["cls_indexes"]. they both are lists and correspond to class name and class id respectively. 

# In[ ]:


# construct annotations. 
from tqdm import tqdm
import json
import os
from pycocotools.mask import encode, decode, area


def dumpJsontoFolder(filename, dataset: dict):
	"""cannot directly save in /disk6 because of import and permission conflict."""
	annFileName = filename
	with open(os.path.join(config.custom_output_path, "annotations", annFileName), 'w') as f:
		json.dump(dataset, f)


def constructAnnotationsFolder(filename, subsetInfos, numLimit=-1):
	"""subset can be ycb_trainval_imageInfos, 
					ycb_train_imageInfos, 
					ycb_val_imageInfos. 
	   if numLimit == -1, ignore it. 
	"""
	
	dataset = {"images": [], "annotations": [], "categories": getAllClassesAndIds()}  # this is a json, after all.
	
	with tqdm(total=len(subsetInfos)) as pbar:
		cnt = 0
		for i_img, info in enumerate(subsetInfos):
			### image related. 
			imageAnn = {}
			
			imgPath = getImageColorPath(info)
			img = mp.imread(imgPath)
			imageAnn["file_name"] = imageId2imageName(info["cocoId"])
			imageAnn["height"] = img.shape[0]
			imageAnn["width"] = img.shape[1]
			imageAnn["id"] = info["cocoId"]
			dataset["images"].append(imageAnn)
			
			### class related has been acquired. 
			
			### annotation related. 
			segPath = getSegMaskPath(info)
			metaPath = getMetaMatPath(info)
			bboxPath = getBboxPath(info)
			seg = mp.imread(segPath)
			meta = loadmat(metaPath)
			bboxes = loadBboxes(bboxPath)
			centers = meta['center']
			clsIds = meta['cls_indexes'].flatten()
			
			for i in range(len(bboxes)):
				annoAnn = {}
				annoAnn["image_id"] = imageAnn["id"]  # be consistent with above. 
				annoAnn["bbox"] = bboxes[i]["bbox"]
				annoAnn["iscrowd"] = 1
				annoAnn["id"] = len(dataset["annotations"])
				# use int to convert ndarray to ensure serializable. 
				annoAnn["category_id"] = int(clsIds[i])
				
				# create segmentation. 
				height, width = seg.shape[0], seg.shape[1]
				center = centers[i]
				# make sure center is within image. 
				center = (min(int(center[0]), width - 1), min(int(center[1]), height - 1))
				segVal = seg[int(center[1])][int(center[0])]
				segMask = np.zeros_like(seg, dtype=np.uint8)
				segMask[seg != segVal] = 0
				segMask[seg == segVal] = 1
				segMask = np.asfortranarray(segMask)
				rle = encode(segMask)
				# byte and string are handled differently in python 3.
				# decode bytes to string, then dump to json. 
				rle['counts'] = rle['counts'].decode('ascii')
				annoAnn["segmentation"] = rle
				annoAnn["area"] = int(area(rle))
				
				dataset["annotations"].append(annoAnn)
			
			pbar.update(1)
			cnt += 1
			if cnt == numLimit:
				break
	dumpJsontoFolder(filename, dataset)


if config.create_trainval:
	constructAnnotationsFolder("ycb_instances_trainval.json", ycb_trainval_imageInfos)
if config.create_train:
	constructAnnotationsFolder("ycb_instances_train.json", ycb_train_imageInfos)
if config.create_val:
	constructAnnotationsFolder("ycb_instances_val.json", ycb_val_imageInfos)
