# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

lib_path1 = osp.join(this_dir, 'options')
add_path(lib_path1)

lib_path2 = osp.join(this_dir, 'mains')
add_path(lib_path2)

lib_path2 = osp.join(this_dir, 'models')
add_path(lib_path2)

lib_path2 = osp.join(this_dir, 'trainers')
add_path(lib_path2)

lib_path2 = osp.join(this_dir, 'utils')
add_path(lib_path2)
# def init_path():
#     this_dir = osp.dirname(__file__)
#
#     lib_path1 = osp.join(this_dir, 'options')
#     add_path(lib_path1)