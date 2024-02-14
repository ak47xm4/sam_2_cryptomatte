# import binascii
import mmh3
import numpy as np
import sys

# np.set_printoptions(threshold=sys.maxsize)  # dont skip print

import struct


def mm3hash_float(name):
    hash_32 = mmh3.hash(name)
    exp = hash_32 >> 23 & 255
    if (exp == 0) or (exp == 255):
        hash_32 ^= 1 << 23

    packed = struct.pack('<L', hash_32 & 0xffffffff)
    return struct.unpack('<f', packed)[0]


def id_to_hex(id):
    return "{0:08x}".format(struct.unpack('<I', struct.pack('<f', id))[0])


def layer_hash(layer_name):
    return id_to_hex(mm3hash_float(layer_name))[:-1]


def hash_object_name(object_name):

    float_value = mm3hash_float(object_name)

    hex_fk = id_to_hex(mm3hash_float(object_name))

    return {
        'hash_hex': hex_fk,
        'fff': float_value,
        'fk': {
            object_name: hex_fk
        }
    }
