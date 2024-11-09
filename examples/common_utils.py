import logging
import math
import os
from datetime import datetime


def setup_logger(main_logger, filename):
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(filename)
    fileHandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    main_logger.addHandler(ch)
    main_logger.addHandler(fileHandler)


def char_map(index):
    if index < 26:
        return index + 97
    elif index < 52:
        return index - 26 + 65
    else:
        return index - 52 + 48


def unique_string(n):
    """generate unique n-length string

    n: length of string
    """
    from functools import reduce

    if n == 0:
        return ""
    byte_len = math.ceil(math.log2(52) + math.log2(62) * (n - 1))
    num = reduce(lambda x, y: x * 256 + y, os.urandom(byte_len), 0)
    codes = []
    codes.append(char_map(num % 52))
    num = math.floor(num / 52)
    for i in range(1, n):
        codes.append(char_map(num % 62))
        num = math.floor(num / 62)

    return "".join(map(chr, codes))


def get_experiment_id(n):
    expid = datetime.now().strftime("%Y-%m-%d-%H:%M:%S-") + unique_string(n)
    return expid
