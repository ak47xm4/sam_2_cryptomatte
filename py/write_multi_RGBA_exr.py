import cv2
import Imath
import OpenEXR
import numpy as np
import json


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

    header['cryptomatte/3ae39a5/conversion'] = b"uint32_to_float32"
    header['cryptomatte/3ae39a5/manifest'] = json.dumps(manifest_data).encode(
        'utf-8')
    header['cryptomatte/3ae39a5/name'] = b"ViewLayer.CryptoObject"

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
        img_float = img.astype(np.float32)
        for i, channel in enumerate("RGBA"):
            pixel_data[f"{layer_name}.{channel}"] = img_float[:, :,
                                                              i].tobytes()

    # Write pixels
    exr_file.writePixels(pixel_data)
    exr_file.close()
