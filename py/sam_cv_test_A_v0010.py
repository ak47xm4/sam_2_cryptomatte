import cv2
import numpy as np
# import OpenEXR
# import Imath

from mijo_fk_cv_np import *
from mmh_test import string_to_cm_float
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

cm0 = np.zeros((height, width, 4), dtype=np.float32)
cm0_r = np.zeros((height, width, 1), dtype=np.float32)
cm0_g = np.zeros((height, width, 1), dtype=np.float32)

bool_depth_1 = (img_sum == (1 / 255))

# print(np.sum(bool_depth_1[bool_depth_1 == True]))

for key, value in images.items():
    bool_png_Mask = (value > 0)
    intersection = np.logical_and(bool_png_Mask, bool_depth_1)
    cm_value = string_to_cm_float(key)
    # print(value)
    # 將符合條件的位置上色為紅色 (0, 0, 255)，可以自行更改顏色
    indices = np.where(intersection)
    indices = indices[:2]  # Select only the first two arrays
    cm0_r[indices] = cm_value
    # cm0[:, :, 0][indices] = value

cm0_g += 1

cm0[:, :, 0] = np.squeeze(cm0_r)
# cm0[:, :, 1] = np.squeeze(cm0_g)

# iiii = [img_sum, img_sum, img_sum]

fk_RGBA = np.zeros((height, width, 4), dtype=np.float32)
iiii = [fk_RGBA, cm0, cm0, cm0]
# 保存为多层EXR文件
output_file = 'output.exr'
layer_names = [
    'ViewLayer', 'ViewLayer.CryptoObject00', 'ViewLayer.CryptoObject01',
    'ViewLayer.CryptoObject02'
]  # Unique layer names

# 設定Cryptomatte元數據
manifest_data = {
    "objectName1": "hash1",
    "objectName2": "hash2",
    # 添加更多物體和對應的hash值
}

metadata = {"name": "sam", "age": 18}

save_multi_layer_exr(output_file, iiii, layer_names, metadata, manifest_data)
