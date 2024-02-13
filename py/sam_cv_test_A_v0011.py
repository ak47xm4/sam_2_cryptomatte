import cv2
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)  # dont skip print

# import OpenEXR
# import Imath

from mijo_fk_cv_np import *
from mmh_test import *
from write_multi_RGBA_exr import *

import os

# Define the directory where the PNG files are located
directory = r'C:\Users\Administrator\Documents\GitHub\sam_2_cryptomatte\test_mask_png'

# Create an empty dictionary
images = {}
# images_list = []

# Loop through the files in the directory
for filename in os.listdir(directory):
    # If the file is a PNG file
    if filename.endswith('.png'):
        # Read the image using cv2.imread() and store it in the dictionary
        filename_str = filename.replace('.png', '')  # can better

        img_2_flt32 = cv2.imread(os.path.join(directory, filename),
                                 cv2.IMREAD_UNCHANGED)
        img_2_flt32 = np.float32(img_2_flt32) / 255.0
        images[filename_str] = red_channel_to_alpha(img_2_flt32)

# for i in images:
# images_list.append(images[i])

height, width, _ = images['0'].shape
######################################################

random_integers = np.random.randint(0, 5, size=(height, width), dtype=np.uint8)

all_mask_sum = merge_sum_images(images)

# save_as_exr(all_mask_sum, 'merged_image.exr')

img_sum = all_mask_sum

# test not offical
cm0 = np.zeros((height, width, 4), dtype=np.float32)
cm0_r = np.zeros((height, width), dtype=np.float32)
cm0_g = np.zeros((height, width), dtype=np.float32)
cm0_b = np.zeros((height, width), dtype=np.float32)
cm0_a = np.zeros((height, width), dtype=np.float32)

cm1 = np.zeros((height, width, 4), dtype=np.float32)
cm1_r = np.zeros((height, width), dtype=np.float32)
cm1_g = np.zeros((height, width), dtype=np.float32)
cm1_b = np.zeros((height, width), dtype=np.float32)
cm1_a = np.zeros((height, width), dtype=np.float32)

cm2 = np.zeros((height, width, 4), dtype=np.float32)
cm2_r = np.zeros((height, width), dtype=np.float32)
cm2_g = np.zeros((height, width), dtype=np.float32)
cm2_b = np.zeros((height, width), dtype=np.float32)
cm2_a = np.zeros((height, width), dtype=np.float32)

bool_depth_1 = ((np.abs(1 - (img_sum * 255))) < 0.1)
bool_depth_2 = ((np.abs(2 - (img_sum * 255))) < 0.1)
bool_depth_3 = ((np.abs(3 - (img_sum * 255))) < 0.1)
bool_depth_4 = ((np.abs(4 - (img_sum * 255))) < 0.1)
bool_depth_5 = ((np.abs(5 - (img_sum * 255))) < 0.1)
bool_depth_6 = ((np.abs(6 - (img_sum * 255))) < 0.1)

manifest_data = {}
# cm 1
for key, value in images.items():
    # the mask from overlapping
    bool_png_Mask = (value > 0)
    # let bool value been useful
    float_png_Mask = bool_png_Mask.astype(np.float32)
    intersection = np.logical_and(bool_png_Mask, bool_depth_1)
    # cm_value = string_to_cm_float(key)
    aaa = hash_object_name(key)
    cm_value = aaa['fff']
    # print(cm_value)
    # print(value)
    # 將符合條件的位置上色為紅色 (0, 0, 255)，可以自行更改顏色
    indices = np.where(intersection)
    indices = indices[:2]  # Select only the first two arrays
    cm0_r[indices] = cm_value
    cm0_g[indices] = 255
    manifest_data.update(aaa['fk'])
cm0[:, :, 0] = cm0_r
cm0[:, :, 1] = cm0_g
# cm 1

# cm 2_1
for key, value in images.items():
    # the mask from overlapping
    bool_png_Mask = (value > 0)
    # rand area
    # print(bool_rand)
    float_png_Mask = bool_png_Mask.astype(np.float32)
    intersection = np.logical_and(bool_png_Mask, bool_depth_2)
    # cm_value = string_to_cm_float(key)
    aaa = hash_object_name(key)

    sss = np.sum(intersection)
    # print(sss)

    cm_value = aaa['fff']

    # 將符合條件的位置上色為紅色 (0, 0, 255)，可以自行更改顏色
    indices = np.where(intersection)
    indices = indices[:2]  # Select only the first two arrays
    cm0_b[indices] = cm_value
    cm0_a[indices] = 255
# cm0[:, :, 2] = cm0_b
# cm0[:, :, 3] = cm0_a
# cm 2_1
# print(cm2_overlapping_key)
# cm 2_2
for key, value in images.items():
    # the mask from overlapping
    bool_png_Mask = (value > 0)
    # rand area
    bool_rand = (random_integers > 3)
    cm_value = aaa['fff']
    # print(bool_rand)
    # float_png_Mask = bool_png_Mask.astype(np.float32)
    intersection_1 = np.logical_and(bool_png_Mask, bool_depth_2)
    bool_replace = (cm0_b != cm_value)
    intersection_2 = np.logical_and(intersection_1[:, :, 0], bool_replace)

    intersection = np.logical_and(intersection_2, bool_rand)
    # cm_value = string_to_cm_float(key)
    aaa = hash_object_name(key)

    # 將符合條件的位置上色為紅色 (0, 0, 255)，可以自行更改顏色
    indices = np.where(intersection)
    indices = indices[:2]  # Select only the first two arrays
    cm0_b[indices] = cm_value
    cm0_a[indices] = 255
cm0[:, :, 2] = cm0_b
cm0[:, :, 3] = cm0_a

# cm1[:, :, 0] = cm1_r
# cm1[:, :, 1] = cm1_g

# fk_RGBA = np.zeros((height, width, 4), dtype=np.float32)
iiii = [cm0, cm1, cm2]
# 保存为多层EXR文件
output_file = 'output.exr'
layer_names = [
    'ViewLayer.CryptoObject00', 'ViewLayer.CryptoObject01',
    'ViewLayer.CryptoObject02'
]  # Unique layer names

# 設定Cryptomatte元數據

metadata = {"name": "sam", "age": 18}  # fuck

save_multi_layer_exr(output_file, iiii, layer_names, metadata, manifest_data)
