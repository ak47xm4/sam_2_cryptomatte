# import cv2
import Imath
import OpenEXR
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)  # dont skip print

import json
from mmh_test import *


def save_multi_layer_exr(output_path, images, layer_names, metadata,
                         manifest_data):
    """
    Write multiple RGBA images to a single EXR file with multiple layers.
    images: List of NumPy arrays of image data
    layer_names: List of layer names for each image
    """
    if not images:
        return  # No images to save

    # Assume all images have the same resolution
    height, width, _ = images[0].shape
    header = OpenEXR.Header(width, height)
    float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = dict()
    # header['cryptomatte'] = b'aaa'  # Enable Cryptomatte support
    # header['fuck'] = {"fuck": "qwerty"}
    # header['channels'] = dict([(c, float_chan) for c in "RGBA"])

    # layer_hash_str = layer_hash('ViewLayer.CryptoObject')
    layer_hash_str = layer_hash('ViewLayer.CryptoObject')

    print(f"fuuuuuuuuk:{layer_hash_str}")

    header['cryptomatte/' + layer_hash_str +
           '/conversion'] = b"uint32_to_float32"
    header['cryptomatte/' + layer_hash_str +
           '/manifest'] = json.dumps(manifest_data).encode('utf-8')
    header['cryptomatte/' + layer_hash_str +
           '/name'] = b"ViewLayer.CryptoObject"
    header['cryptomatte/' + layer_hash_str + '/hash'] = b"Murmurhash3_32"

    # header['metadata'] = metadata

    for img, layer_name in zip(images, layer_names):
        # Convert image to half float
        img_float = img.astype(np.float32)

        # Prepare the channels for this layer
        for i, channel in enumerate("RGBA"):
            header['channels'][f"{layer_name}.{channel}"] = float_chan

    # Create the EXR file
    exr_file = OpenEXR.OutputFile(output_path, header)

    # 将元数据写入图像
    # exr_file.add_header(metadata)

    # Prepare pixel data
    pixel_data = dict()
    for img, layer_name in zip(images, layer_names):
        img /= 255
        img_float = img.astype(np.float32)
        for i, channel in enumerate("RGBA"):
            px_value = img_float[:, :, i].tobytes()
            pixel_data[f"{layer_name}.{channel}"] = px_value
            # if channel == "R" and layer_name == "ViewLayer.CryptoObject00":
            # recovered_arr = np.frombuffer(px_value, dtype=np.float32)
            # print(recovered_arr)
    # Write pixels
    exr_file.writePixels(pixel_data)
    # print(dir(exr_file.writePixels))
    exr_file.close()
