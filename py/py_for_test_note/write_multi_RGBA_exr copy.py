import cv2
import Imath, OpenEXR
import numpy as np


def save_multi_layer_exr(output_path, images, layer_names):
    """
    Write multiple RGBA images to a single EXR file with multiple layers.
    images: List of image paths
    layer_names: List of layer names for each image
    """
    header = OpenEXR.Header(images[0].shape[0], images[0].shape[1])
    channels = {}
    print('aaaaaaaaaaaaa')

    for img_path, layer_name in zip(images, layer_names):
        print('bbbbbbbbbbb')
        # img = cv2.imread(img_path,
        #  cv2.IMREAD_UNCHANGED)  # Read image with alpha channel
        img = img_path
        height, width, _ = img.shape

        # Update header size for EXR
        # header['dataWindow'] = Imath.Box2i(Imath.V2i(0, 0),
        #    Imath.V2i(width - 1, height - 1))

        # Convert image to half float
        img_half = img.astype(np.float16)

        # Separate channels and add to EXR channels
        for i, channel in enumerate("RGBA"):
            channel_data = img_half[:, :, i].tobytes()
            channels[f"{layer_name}.{channel}"] = OpenEXR.Channel(
                Imath.PixelType(Imath.PixelType.HALF))
            header['channels'][f"{layer_name}.{channel}"] = channels[
                f"{layer_name}.{channel}"]

    # Create EXR file and write channels
    exr_file = OpenEXR.OutputFile(output_path, header)
    exr_file.writePixels(channels)
    exr_file.close()
