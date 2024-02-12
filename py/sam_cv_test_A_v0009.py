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

height, width, _ = images['0'].shape

random_integers = np.random.randint(0, 5, size=(height, width), dtype=np.uint8)

all_mask_sum = merge_sum_images(images)

# save_as_exr(all_mask_sum, 'merged_image.exr')

img = all_mask_sum
# print(all_mask_sum)
# print(img)
'''
color_values = [0, 1, 2, 3, 4, 5]

for color_value in color_values:
    # be easier
    color_values_int = int(color_value)
    color_value /= 255

    # Define the color range
    lower_bound = np.array([color_value] * 4)
    # upper_bound = np.array([color_value + 1] * 4)
    upper_bound = np.array([color_values[-1] / 255] * 4)

    # Find pixels within the color range
    mask = cv2.inRange(img, lower_bound, upper_bound)
    # print(mask)

    # Apply the mask to the image
    result = cv2.bitwise_and(img, img, mask=mask)

    # Save the result
    save_as_exr(result, f'merged_image_{color_values_int}.exr')
'''
# 读取输入图像
# image_paths = ['image1.png', 'image2.png', 'image3.png']  # 你的输入图像路径
# images = [cv2.imread(image_path) for image_path in image_paths]

iiii = [img, img]

# print(iiii[0].shape)

# 保存为多层EXR文件
output_file = 'output.exr'
layer_names = ['Layer1', 'Layer2']  # Unique layer names
save_multi_layer_exr(output_file, iiii, layer_names)