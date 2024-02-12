# aaa = 5

# bbb = 2

# print(aaa // bbb)
import cv2
import numpy as np

# 创建两个示例图像
image1 = np.zeros((300, 300), dtype=np.uint8)
image2 = np.zeros((300, 300), dtype=np.uint8)

# 在第一个图像上绘制一个白色矩形
cv2.rectangle(image1, (50, 50), (250, 250), 255, -1)

# 在第二个图像上绘制一个白色圆形
cv2.circle(image2, (150, 150), 100, 255, -1)

# 对两个图像进行按位与操作
result = cv2.bitwise_and(image1, image2)

# 显示结果
cv2.imshow('Image 1', image1)
cv2.imshow('Image 2', image2)
cv2.imshow('Bitwise AND Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
