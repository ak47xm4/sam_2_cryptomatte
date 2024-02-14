# aaa = 5

# bbb = 2

# print(aaa // bbb)
# import cv2
# import numpy as np

# # 创建两个示例图像
# image1 = np.zeros((300, 300), dtype=np.uint8)
# image2 = np.zeros((300, 300), dtype=np.uint8)

# # 在第一个图像上绘制一个白色矩形
# cv2.rectangle(image1, (50, 50), (250, 250), 255, -1)

# # 在第二个图像上绘制一个白色圆形
# cv2.circle(image2, (150, 150), 100, 255, -1)

# # 对两个图像进行按位与操作
# result = cv2.bitwise_and(image1, image2)

# # 显示结果
# cv2.imshow('Image 1', image1)
# cv2.imshow('Image 2', image2)
# cv2.imshow('Bitwise AND Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import cv2
import OpenEXR
import Imath
import numpy as np


def save_multi_layer_exr(images, output_file):
    # 创建一个OpenEXR文件对象
    exr_header = OpenEXR.Header(images[0].shape[1], images[0].shape[0])
    # 设置通道类型
    exr_header['channels'] = dict(
        (('R', Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))),
         ('G', Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))),
         ('B', Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))),
         ('A', Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)))))
    # 添加图像图层
    exr = OpenEXR.OutputFile(output_file, exr_header)

    # 将图像转换为numpy数组并写入EXR文件
    for i, image in enumerate(images):
        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        r = image_rgba[:, :, 0].astype(np.float32)
        g = image_rgba[:, :, 1].astype(np.float32)
        b = image_rgba[:, :, 2].astype(np.float32)
        a = image_rgba[:, :, 3].astype(np.float32)

        exr.writePixels(
            {
                'R': r.tobytes(),
                'G': g.tobytes(),
                'B': b.tobytes(),
                'A': a.tobytes()
            }, i)

    exr.close()


# 读取输入图像
image_paths = ['image1.png', 'image2.png', 'image3.png']  # 你的输入图像路径
images = [cv2.imread(image_path) for image_path in image_paths]

# 保存为多层EXR文件
output_file = 'output.exr'
save_multi_layer_exr(images, output_file)
