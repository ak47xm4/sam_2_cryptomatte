import cv2
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)  # dont skip print

from mijo_fk_cv_np import *
from mmh3_for_cryptomatte import *
from write_multi_RGBA_exr import *

import os

aaa = "2"

bbb = hash_object_name(aaa)

print(bbb)
