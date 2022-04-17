# -*- coding: utf-8 -*-

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from config import Config
# import utils
import model as modellib, utils
import visualize
import yaml
from model import log
from PIL import Image

import torch
print(yaml.__version__)
"""
加入自己类别名称
更改类别个数
"""

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Root directory of the project
ROOT_DIR = os.getcwd()

# ROOT_DIR = os.path.abspath("../")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

iter_num = 0

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes_0239.pth")
# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 2048
    IMAGE_MAX_DIM = 2048
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16 * 6, 32 * 6, 64 * 6, 128 * 6, 256 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


config = ShapesConfig()
config.display()


class DrugDataset(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的自己的类别
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    # yaml_pathdataset_root_path = "/tongue_dateset/"
    # img_floder = dataset_root_path + "rgb"
    # mask_floder = dataset_root_path + "mask"
    # dataset_root_path = "/tongue_dateset/"
    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        # self.add_class("shapes", 1, "tank") # 黑色素瘤
        #
        # self.add_class("shapes", 1, "Railway")
        self.add_class("shapes", 1, "fastener_L1")
        self.add_class("shapes", 2, "fastener_R1")
        self.add_class("shapes", 3, "hat_L")
        self.add_class("shapes", 4, "hat_R")
        self.add_class("shapes", 5, "shim_L")
        self.add_class("shapes", 6, "shim_R")


        for i in range(count):
            tempszh = imglist[i].split(".")[1]
            # 获取图片宽和高
            if imglist[i].split(".")[1] == 'png':

                filestr = imglist[i].split(".")[0]

                mask_path = mask_floder + "/" + filestr + ".png"
                yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
                print(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
                cv_img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")

                self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                               width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # 重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            # if labels[i].find("Railway") != -1:
            #     labels_form.append("Railway")
            if labels[i].find("fastener_L1") != -1:
                labels_form.append("fastener_L1")
            if labels[i].find("fastener_R1") != -1:
                labels_form.append("fastener_R1")
            if labels[i].find("hat_L") != -1:
                labels_form.append("hat_L")
            if labels[i].find("hat_R") != -1:
                labels_form.append("hat_R")
            if labels[i].find("shim_L") != -1:
                labels_form.append("shim_L")
            if labels[i].find("shim_R") != -1:
                labels_form.append("shim_R")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# 基础设置
dataset_root_path = "mydata/"
img_floder = dataset_root_path + "pic"
mask_floder = dataset_root_path + "cv2_mask"
imglist = os.listdir(img_floder)
count = len(imglist)

model = modellib.MaskRCNN(config=config,
                          model_dir=MODEL_DIR)
if config.GPU_COUNT:
    model = model.cuda()
# train与val数据集准备
dataset_train = DrugDataset()
dataset_train.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
dataset_train.prepare()

print("dataset_train-->", dataset_train._image_ids)

dataset_val = DrugDataset()
dataset_val.load_shapes(6, img_floder, mask_floder, imglist, dataset_root_path)
dataset_val.prepare()

print("dataset_val-->", dataset_val._image_ids)

# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#    image = dataset_train.load_image(image_id)
#    mask, class_ids = dataset_train.load_mask(image_id)
#    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Create model in training mode


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights())
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH)
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1])

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
# model.train_model(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE / 10,
#             epochs=5,
#             layers='2+')
#
# model.train_model(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=100,
#             layers='2+')

# model.train_model(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=180,
#             layers='4+')

# model.train_model(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=160,
#             layers='all')

model.train_model(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=60,
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE / 10,
#             epochs=300,
#             layers="all")
#
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=350,
#             layers='heads')
