import cv2
import Imath
import OpenEXR
import numpy as np


def save_multi_layer_exr(output_path, images, layer_names):
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
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = dict()

    for img, layer_name in zip(images, layer_names):
        # Convert image to half float
        img_float = img.astype(np.float32)

        # Prepare the channels for this layer
        for i, channel in enumerate("RGBA"):
            header['channels'][f"{layer_name}.{channel}"] = half_chan

    # Create the EXR file
    exr_file = OpenEXR.OutputFile(output_path, header)

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
