import cv2
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)  # dont skip print

from mijo_fk_cv_np import *
from mmh3_for_cryptomatte import *
from write_multi_RGBA_exr import *

import os

import time

start_time = time.time()

# Define the directory where the PNG files are located
directory = r'H:\20240213_sam_cm_test\render\sam\PF0003-01_Clip-1.0001'
# directory = r'C:\Users\Administrator\Documents\GitHub\sam_2_cryptomatte\test_mask_png'

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

print(" load img 2 np --- %s seconds ---" % (time.time() - start_time))

random_integers = np.random.randint(0, 6, size=(height, width), dtype=np.uint8)

# test not offical
cm0 = np.zeros((height, width, 4), dtype=np.float32)
cm0_r = np.zeros((height, width), dtype=np.float32)
cm0_g = np.zeros((height, width), dtype=np.bool_)
cm0_b = np.zeros((height, width), dtype=np.float32)
cm0_a = np.zeros((height, width), dtype=np.bool_)

cm1 = np.zeros((height, width, 4), dtype=np.float32)
cm1_r = np.zeros((height, width), dtype=np.float32)
cm1_g = np.zeros((height, width), dtype=np.bool_)
cm1_b = np.zeros((height, width), dtype=np.float32)
cm1_a = np.zeros((height, width), dtype=np.bool_)

cm2 = np.zeros((height, width, 4), dtype=np.float32)
cm2_r = np.zeros((height, width), dtype=np.float32)
cm2_g = np.zeros((height, width), dtype=np.bool_)
cm2_b = np.zeros((height, width), dtype=np.float32)
cm2_a = np.zeros((height, width), dtype=np.bool_)

cm_channel_id_list = [cm0_r, cm0_b, cm1_r, cm1_b, cm2_r, cm2_b]

cm_channel_mask_list = [cm0_g, cm0_a, cm1_g, cm1_a, cm2_g, cm2_a]

print(" create np datas --- %s seconds ---" % (time.time() - start_time))

manifest_data = {}

# 生成Cryptomatte數據

for key, value in images.items():
    # manifest_data
    hash_obj_dict = hash_object_name(key)
    cm_value = hash_obj_dict['fff']
    manifest_data.update(hash_obj_dict['fk'])

    # the mask from overlapping
    bool_png_Mask = (value[:, :, 0] > 0)
    for i in range(0, 6):
        intersection = np.logical_and(bool_png_Mask, cm_channel_mask_list[i])

        all_false = np.all(~intersection)

        # print(all_false)
        if all_false:
            cm_channel_id_list[i][bool_png_Mask] = cm_value
            cm_channel_mask_list[i][bool_png_Mask] = np.array(
                [True])  # Change this line
            break

        intersection_not = np.logical_not(intersection)
        intersection_not = np.logical_and(intersection, bool_png_Mask)

        indices = np.where(intersection_not)
        indices = indices[:2]  # Select only the first two arrays

        cm_channel_id_list[i][indices] = cm_value
        cm_channel_mask_list[i][indices] = True
        bool_png_Mask[indices] = False

print(" 生成Cryptomatte數據 --- %s seconds ---" % (time.time() - start_time))

for i in range(1, 6):
    pass
    rand_area_1 = (random_integers == i)

    indices = rand_area_1

    # rand overlapping pass layer 0
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

print(" 打亂CM --- %s seconds ---" % (time.time() - start_time))

cm0[:, :, 0] = cm_channel_id_list[0]
cm0[:, :, 1] = cm_channel_mask_list[0].astype(np.float32)
cm0[:, :, 2] = cm_channel_id_list[1]
cm0[:, :, 3] = cm_channel_mask_list[1].astype(np.float32)

cm1[:, :, 0] = cm_channel_id_list[2]
cm1[:, :, 1] = cm_channel_mask_list[2].astype(np.float32)
cm1[:, :, 2] = cm_channel_id_list[3]
cm1[:, :, 3] = cm_channel_mask_list[3].astype(np.float32)

cm2[:, :, 0] = cm_channel_id_list[4]
cm2[:, :, 1] = cm_channel_mask_list[4].astype(np.float32)
cm2[:, :, 2] = cm_channel_id_list[5]
cm2[:, :, 3] = cm_channel_mask_list[5].astype(np.float32)

iiii = [cm0, cm1, cm2]
# 保存为多层EXR文件
output_file = 'output.exr'
layer_names = [
    'ViewLayer.CryptoObject00', 'ViewLayer.CryptoObject01',
    'ViewLayer.CryptoObject02'
]  # Unique layer names

# 設定Cryptomatte元數據
save_multi_layer_exr(output_file, iiii, layer_names, manifest_data)

print(" end --- %s seconds ---" % (time.time() - start_time))
