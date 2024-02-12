import mmh3
import numpy as np
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


# 测试
# input_string = "example_string"
# cryptomatte_hash = string_to_cryptomatte_hash(input_string)
# hash_float = string_to_cm_float(input_string)
# print("Float representation of Cryptomatte hash '{}' is: {}".format(
#     cryptomatte_hash, hash_float))
