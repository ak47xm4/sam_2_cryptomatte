import cv2
import numpy as np

# 读取图像文件
image1 = cv2.imread('input.png')
image2 = cv2.imread('input2.png')

def red_channel_to_alpha(image):
    # 分割图像通道
    b, g, r = cv2.split(image)

    # 创建一个具有相同尺寸但只有一个通道的空白图像
    alpha = np.zeros_like(r)

    # 将红色通道值赋值给Alpha通道
    alpha[:, :] = r

    # 合并Alpha通道到原始图像
    return cv2.merge([b, g, r, alpha])  # return the new image

# update the original images with the returned images
image1 = red_channel_to_alpha(image1)
image2 = red_channel_to_alpha(image2)


alpha_image1 = image1[:,:,3] / 255.0
alpha_image2 = image2[:,:,3] / 255.0

# set adjusted colors
for color in range(0, 3):
    image1[:,:,color] = alpha_image2 * image2[:,:,color] + \
        alpha_image1 * image1[:,:,color] * (1 - alpha_image2)

# set adjusted alpha and denormalize back to 0-255
image1[:,:,3] = (1 - (1 - alpha_image2) * (1 - alpha_image1)) * 255

# 保存合并后的图像
cv2.imwrite('merged_image.png', image1)