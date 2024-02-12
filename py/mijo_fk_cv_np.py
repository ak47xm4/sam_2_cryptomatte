import cv2
import numpy as np
import OpenEXR
import Imath


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
        'R': image[:, :, 0],
        'G': image[:, :, 1],
        'B': image[:, :, 2],
        'A': image[:, :, 3]
        # 'R': image[:, :, 0].tobytes(),
        # 'G': image[:, :, 1].tobytes(),
        # 'B': image[:, :, 2].tobytes(),
        # 'A': image[:, :, 3].tobytes()
    })
    exr.close()


def red_channel_to_alpha(image):
    image = np.float32(image) / 255.0
    if len(image.shape) == 2:  # Grayscale image, so convert to 3-channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    b, g, r = cv2.split(image)
    return cv2.merge([b, g, r, r])


def merge_over_images(image1, image2):
    # 轉換圖像到浮點數表示
    image1 = np.float32(image1) / 255.0
    image2 = np.float32(image2) / 255.0

    alpha_image1 = image1[:, :, 3]
    alpha_image2 = image2[:, :, 3]

    # 計算合併後的顏色和透明度
    for color in range(0, 3):
        image1[:, :,
               color] = alpha_image2 * image2[:, :,
                                              color] * 255.0 + alpha_image1 * image1[:, :, color] * (
                                                  1 - alpha_image2) * 255.0

    image1[:, :, 3] = 1 - (1 - alpha_image2) * (1 - alpha_image1)

    return image1


def merge_add_images(image1, image2):
    # 轉換圖像到浮點數表示
    image1 = np.float32(image1) / 255.0
    image2 = np.float32(image2) / 255.0

    # 計算合併後的顏色和透明度
    # for color in range(0, 4):
    # image1[:, :, color] = image1[:, :, color] + image2[:, :, color]

    # OK, this is the correct way to add the images
    # image1 = image1 + image2

    # This is the correct way to add the images
    np.sum([image1, image2], axis=0, out=image1)

    return image1


def merge_sum_images(images_dict):
    image1 = images_dict['0']
    images_list = []

    for i in images_dict:
        images_list.append(images_dict[i])

    np.sum(images_list, axis=0, out=image1)

    return image1
