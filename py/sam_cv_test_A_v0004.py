import cv2
import numpy as np
import OpenEXR
import Imath

def red_channel_to_alpha(image):
    image = np.float32(image) / 255.0
    if len(image.shape) == 2:  # Grayscale image, so convert to 3-channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    b, g, r = cv2.split(image)
    return cv2.merge([b, g, r, r])


def merge_images(image1, image2):
    # 轉換圖像到浮點數表示
    image1 = np.float32(image1) / 255.0
    image2 = np.float32(image2) / 255.0

    alpha_image1 = image1[:, :, 3]
    alpha_image2 = image2[:, :, 3]

    # 計算合併後的顏色和透明度
    for color in range(0, 3):
        image1[:, :, color] = alpha_image2 * image2[:, :, color]  * 255.0 + alpha_image1 * image1[:, :, color] * (1 - alpha_image2) * 255.0

    image1[:, :, 3] = 1 - (1 - alpha_image2) * (1 - alpha_image1)

    return image1

# 讀取並處理圖像
image1 = cv2.imread('input.png', cv2.IMREAD_UNCHANGED)
image2 = cv2.imread('input2.png', cv2.IMREAD_UNCHANGED)

image1 = red_channel_to_alpha(image1)
image2 = red_channel_to_alpha(image2)

merged_image = merge_images(image1, image2)

# 將合併後的圖像轉換為 OpenEXR 需要的格式並保存
def save_as_exr(image, filename):
    # 將數據類型轉換回適合 OpenEXR 的格式
    image = (image * 255).astype(np.float32)
    (height, width, _) = image.shape

    header = OpenEXR.Header(width, height)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = dict([(c, half_chan) for c in "RGBA"])

    exr = OpenEXR.OutputFile(filename, header)
    exr.writePixels({
        'R': image[:, :, 0].tobytes(),
        'G': image[:, :, 1].tobytes(),
        'B': image[:, :, 2].tobytes(),
        'A': image[:, :, 3].tobytes()
    })
    exr.close()

save_as_exr(merged_image, 'merged_image.exr')
