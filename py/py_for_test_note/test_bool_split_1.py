import cv2
import numpy as np

# 假設這是你的布林值numpy陣列
bool_array_1 = np.array([[True, False, True], [False, True, False],
                         [True, False, True]])

# 建立一個300x300的空白影像
image = np.zeros((3, 3, 3), dtype=np.uint8)

# 將符合條件的位置上色為紅色 (0, 0, 255)，可以自行更改顏色
indices = np.where(bool_array_1)
image[indices] = (0, 0, 255)

# 顯示圖片
cv2.imshow('Masked Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
