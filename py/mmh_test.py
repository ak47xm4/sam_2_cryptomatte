import binascii
import mmh3
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)  # dont skip print

import zlib
import struct


def string_to_cryptomatte_hash(string):
    # 使用MurmurHash算法计算字符串的哈希值
    hash_value = mmh3.hash64(string, seed=0)[0]

    # 将哈希值压缩为4字节并添加CRC32校验
    packed_data = struct.pack('>Q', hash_value)
    crc32 = zlib.crc32(packed_data)
    packed_data += struct.pack('>I', crc32 & 0xFFFFFFFF)

    # 返回Cryptomatte哈希值的字符串表示
    return packed_data.hex()


def cryptomatte_hash_to_float(cryptomatte_hash):
    # 将十六进制字符串转换为整数
    hash_int = int(cryptomatte_hash, 16)

    # 将整数转换为 float
    hash_float = float(hash_int)

    return hash_float


# def string_to_cm_float(string):
# cryptomatte_hash = string_to_cryptomatte_hash(string)
# return cryptomatte_hash_to_float(cryptomatte_hash)


def string_to_cm_float(name):
    return np.float32(mmh3.hash(name, 0) & 0xffffffff) / np.float32(2**32)


'''
def mm3hash_float(name):
    hash_32 = mmh3.hash(name)
    exp = hash_32 >> 23 & 255
    if (exp == 0) or (exp == 255):
        hash_32 ^= 1 << 23
    packed = struct.pack('<L', hash_32 & 0xffffffff)
    return struct.unpack('<f', packed)[0]
'''


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
    # 计算 MurmurHash3_32 位哈希
    hash_value = mmh3.hash(object_name.encode('utf-8'), seed=0)

    # 将 uint32 转换为 float32
    # float_value = struct.unpack('f', struct.pack('I',
    #  hash_value & 0xffffffff))[0]
    float_value = mm3hash_float(object_name)

    # hash_hex = struct.unpack('=f', object_name.decode('hex'))[0]
    # hash_hex = mm3_float_2_hex8(float_value)
    hex_fk = id_to_hex(mm3hash_float(object_name))
    # print(hex_fk)
    # print(seven_digit_hash)

    return {
        'hash_hex': hex_fk,
        'fff': float_value,
        'fk': {
            object_name: hex_fk
        }
    }


# aaa = id_to_hex(mm3hash_float('Cube'))

# print(aaa)

# aaa = mm3_str_2_hex8('Cube')
# print(aaa)
# 测试
# object_name = "your_object_name_here"
# hashed_value = hash_object_name(object_name)
# print("哈希值:", hashed_value)

# 测试
# input_string = "example_string"
# cryptomatte_hash = string_to_cryptomatte_hash(input_string)
# hash_float = string_to_cm_float(input_string)
# print("Float representation of Cryptomatte hash '{}' is: {}".format(
#     cryptomatte_hash, hash_float))
