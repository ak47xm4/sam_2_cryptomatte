import cv2
import numpy as np
# import OpenEXR
# import Imath

from mijo_fk_cv_np import *

# 讀取並處理圖像
image1 = cv2.imread('input.png', cv2.IMREAD_UNCHANGED)
# image2 = cv2.imread('input2.png', cv2.IMREAD_UNCHANGED)

image1 = red_channel_to_alpha(image1)

# 获取图像的形状
height, width, _ = image1.shape

# 创建一个与图像大小相同的数组，用于存储字符串变量
string_vars = np.empty((height, width), dtype='object')

# 在每个像素上添加字符串变量
# for i in range(height):
#     for j in range(width):
#         # string_vars[i, j] = "Pixel at ({}, {})".format(i, j)
#         string_vars[i, j] = np.random.randint(0, 5)

# 现在，string_vars 中的每个元素都是一个字符串，表示相应像素的信息。

random_integers = np.random.randint(0, 5, size=(height, width), dtype=np.uint8)

# 你可以通过以下方式访问特定像素处的字符串变量：
print(random_integers)

# save_as_exr(merged_image, 'merged_image.exr')
