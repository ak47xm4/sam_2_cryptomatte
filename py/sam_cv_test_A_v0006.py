import cv2
import numpy as np
# import OpenEXR
# import Imath

from mijo_fk_cv_np import *

import os

# Define the directory where the PNG files are located
directory = r'C:\Users\Administrator\Documents\GitHub\sam_2_cryptomatte\test_mask_png'

# Create an empty dictionary
images = {}

# Loop through the files in the directory
for filename in os.listdir(directory):
    # If the file is a PNG file
    if filename.endswith('.png'):
        # Read the image using cv2.imread() and store it in the dictionary
        filename_str = filename.replace('.png', '')  # can better
        images[filename_str] = cv2.imread(os.path.join(directory, filename),
                                          cv2.IMREAD_UNCHANGED)

height, width = images['0'].shape

random_integers = np.random.randint(0, 5, size=(height, width), dtype=np.uint8)
