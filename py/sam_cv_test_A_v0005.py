import cv2
import numpy as np
import OpenEXR
import Imath

from mijo_fk_cv_np import *

# 讀取並處理圖像
image1 = cv2.imread('input.png', cv2.IMREAD_UNCHANGED)
image2 = cv2.imread('input2.png', cv2.IMREAD_UNCHANGED)

image1 = red_channel_to_alpha(image1)
image2 = red_channel_to_alpha(image2)

# merged_image = merge_over_images(image1, image2)
merged_image = merge_add_images(image1, image2)



save_as_exr(merged_image, 'merged_image.exr')
