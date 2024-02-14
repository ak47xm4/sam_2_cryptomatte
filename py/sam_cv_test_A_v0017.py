import cv2
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)  # dont skip print

from mijo_fk_cv_np import *
from mmh3_for_cryptomatte import *
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
######################################################

random_integers = np.random.randint(0, 6, size=(height, width), dtype=np.uint8)

all_mask_sum = merge_sum_images(images)

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

cm_channel_id_list = [cm0_r, cm0_b, cm1_r, cm1_b, cm2_r, cm2_b]

cm_channel_mask_list = [cm0_g, cm0_a, cm1_g, cm1_a, cm2_g, cm2_a]

bool_depth_1 = ((np.abs(1 - (img_sum * 255))) < 0.1)
bool_depth_2 = ((np.abs(2 - (img_sum * 255))) < 0.1)
bool_depth_3 = ((np.abs(3 - (img_sum * 255))) < 0.1)
bool_depth_4 = ((np.abs(4 - (img_sum * 255))) < 0.1)
bool_depth_5 = ((np.abs(5 - (img_sum * 255))) < 0.1)
bool_depth_6 = ((np.abs(6 - (img_sum * 255))) < 0.1)

bool_depth_list = [
    bool_depth_1, bool_depth_2, bool_depth_3, bool_depth_4, bool_depth_5,
    bool_depth_6
]

manifest_data = {}

# 生成Cryptomatte數據
for i in range(0, 6):
    bool_depth = bool_depth_list[i]
    for key, value in images.items():
        # the mask from overlapping
        bool_png_Mask = (value > 0)
        float_png_Mask = bool_png_Mask.astype(np.float32)
        intersection = np.logical_and(bool_png_Mask, bool_depth)
        hash_obj_dict = hash_object_name(key)

        indices = np.where(intersection)
        indices = indices[:2]  # Select only the first two arrays
        cm_value = hash_obj_dict['fff']

        cm_channel_id_list[i][indices] = cm_value
        cm_channel_mask_list[i][indices] = 1

        # manifest_data
        if (i == 0):
            manifest_data.update(hash_obj_dict['fk'])

# sort by cm layer intersection

for i in range(2, 6):
    pass
    # xor pass layer 0
    xor = np.logical_xor(cm_channel_mask_list[0], cm_channel_mask_list[i])
    intersection_xor = np.logical_and(cm_channel_mask_list[0], xor)
    indices = np.logical_not(intersection_xor)

    temp_array_1 = cm_channel_id_list[0].copy()

    np.copyto(cm_channel_id_list[0],
              cm_channel_id_list[i],
              casting='same_kind',
              where=indices)
    np.copyto(cm_channel_id_list[i],
              temp_array_1,
              casting='same_kind',
              where=indices)

    temp_array_1 = cm_channel_mask_list[0].copy()

    np.copyto(cm_channel_mask_list[0],
              cm_channel_mask_list[i],
              casting='same_kind',
              where=indices)
    np.copyto(cm_channel_mask_list[i],
              temp_array_1,
              casting='same_kind',
              where=indices)
    pass
'''
for i in range(2, 6):
    pass
    # xor pass layer 1
    xor = np.logical_xor(cm_channel_mask_list[1], cm_channel_mask_list[i])
    intersection_xor = np.logical_and(cm_channel_mask_list[1], xor)
    indices = np.logical_not(intersection_xor)

    np.copyto(cm_channel_id_list[1],
              cm_channel_id_list[i],
              casting='same_kind',
              where=indices)

    np.copyto(cm_channel_mask_list[1],
              cm_channel_mask_list[i],
              casting='same_kind',
              where=indices)
'''
'''
# sort cm random overlapping process
for i in range(5, 6):
    pass
    print(i)
    pass

    rand_area_1 = (random_integers == i)

    # rand overlapping pass layer 0
    interesct = np.logical_and(cm_channel_mask_list[0],
                               cm_channel_mask_list[i])
    # indices = np.logical_and(rand_area_1, interesct)
    indices = interesct

    temp_array_1 = cm_channel_id_list[0].copy()

    np.copyto(cm_channel_id_list[0],
              cm_channel_id_list[i],
              casting='same_kind',
              where=indices)
    np.copyto(cm_channel_id_list[i],
              temp_array_1,
              casting='same_kind',
              where=indices)

    # temp_array_1 = cm_channel_mask_list[0].copy()

    # np.copyto(cm_channel_mask_list[0],
    #           cm_channel_mask_list[i],
    #           casting='same_kind',
    #           where=indices)
    # np.copyto(cm_channel_mask_list[i],
    #           temp_array_1,
    #           casting='same_kind',
    #           where=indices)

    pass

    # rand overlapping pass layer 1
    interesct = np.logical_and(cm_channel_mask_list[1],
                               cm_channel_mask_list[i])
    indices = np.logical_and(rand_area_1, interesct)
    temp_array_1 = cm_channel_id_list[1].copy()

    np.copyto(cm_channel_id_list[1],
              cm_channel_id_list[i],
              casting='same_kind',
              where=indices)
    np.copyto(cm_channel_id_list[i],
              temp_array_1,
              casting='same_kind',
              where=indices)

    temp_array_1 = cm_channel_mask_list[1].copy()

    np.copyto(cm_channel_mask_list[1],
              cm_channel_mask_list[i],
              casting='same_kind',
              where=indices)
    np.copyto(cm_channel_mask_list[i],
              temp_array_1,
              casting='same_kind',
              where=indices)
    pass
    pass
'''
cm0[:, :, 0] = cm_channel_id_list[0]
cm0[:, :, 1] = cm_channel_mask_list[0]
cm0[:, :, 2] = cm_channel_id_list[1]
cm0[:, :, 3] = cm_channel_mask_list[1]

cm1[:, :, 0] = cm_channel_id_list[2]
cm1[:, :, 1] = cm_channel_mask_list[2]
cm1[:, :, 2] = cm_channel_id_list[3]
cm1[:, :, 3] = cm_channel_mask_list[3]

cm2[:, :, 0] = cm_channel_id_list[4]
cm2[:, :, 1] = cm_channel_mask_list[4]
cm2[:, :, 2] = cm_channel_id_list[5]
cm2[:, :, 3] = cm_channel_mask_list[5]

iiii = [cm0, cm1, cm2]
# 保存为多层EXR文件
output_file = 'output.exr'
layer_names = [
    'ViewLayer.CryptoObject00', 'ViewLayer.CryptoObject01',
    'ViewLayer.CryptoObject02'
]  # Unique layer names

# 設定Cryptomatte元數據
save_multi_layer_exr(output_file, iiii, layer_names, manifest_data)
