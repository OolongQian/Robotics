#!/usr/bin/env python
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--YCB_dataset_path', type=str, default='datasets/YCB_Video_Dataset', help='YCB dataset root dir')
parser.add_argument('--custom_output_path', type=str, default='datasets/Custom_YCB_Video_Dataset',
                    help='output Custom YCB dataset root dir')
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


# trainval.txt is the union of train.txt and val.txt.
ycb_trainval_index = "trainval.txt"
ycb_trainval_imageInfos = createInfo4Datasubset(ycb_trainval_index)


def imageIds2idSet(imageIds):
	ids = []
	for img_info in imageIds:
		ids.append(img_info['cocoId'])
	return ids


ycb_trainval_idSet = imageIds2idSet(ycb_trainval_imageInfos)

# browse images find, rename, and copy them into images folder.
# typical coco image name could be '000000000139'. 

# use the first image info as an example. 

import os
import matplotlib.image as mp
import matplotlib.pyplot as plt


def getImageColorPath(imgInfo):
	"""get rgb image file name/path"""
	
	def getImageBasePath(imgInfo):
		basePath = os.getcwd()
		ycbPath = os.path.join(basePath, config.YCB_dataset_path)
		basePath = os.path.join(ycbPath, "data", imgInfo["folder"], imgInfo["name"])
		return basePath
	
	basePath = getImageBasePath(imgInfo)
	folderPath, filename = os.path.split(basePath)
	return os.path.join(folderPath, filename + "-color.png")


def plotImage(img):
	plt.imshow(img)
	plt.show()


# save to path.
# - Custom_YCB_Video_Dataset
#     - annotations.
#     - images. 
def saveImageToCustomYCB(info, img):
	"""need to create directory in advance."""
	base_path = os.getcwd()
	customYcbPath = os.path.join(base_path, config.custom_output_path)
	# typical coco image name has 12 decimals.
	filename = "".join(["0" for _ in range(12 - len(str(info["cocoId"])))]) + str(info["cocoId"]) + ".jpg"
	filename = os.path.join(customYcbPath, "images", filename)
	mp.imsave(filename, img)


# construct custom /images folder.
from tqdm import tqdm


def constructImagesFolder(numLimit=-1):
	with tqdm(total=len(ycb_trainval_imageInfos)) as pbar:
		for i, info in enumerate(ycb_trainval_imageInfos):
			pngPathSample = getImageColorPath(info)
			img = mp.imread(pngPathSample)
			saveImageToCustomYCB(info, img)
			pbar.update(1)
			if i == numLimit:
				break


constructImagesFolder()
