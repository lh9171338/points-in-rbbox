# -*- encoding: utf-8 -*-
"""
@File    :   setup.py
@Time    :   2025/04/18 16:30:05
@Author  :   lh9171338
@Version :   1.0
@Contact :   2909171338@qq.com
"""

import os
import re
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_version():
    """get version"""
    with open("points_in_rbbox/__init__.py", "r") as f:
        content = f.read()
    version_match = re.search(r'^__version__ = ["\']([^"\']+)["\']', content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("无法解析 __version__")


if os.getenv("ENABLE_BF16", "0") == "1":
    print("BF16 is enabled")
    cxx_args = ["-g", "-DENABLE_BF16"]
    nvcc_args = ["-O2", "-DENABLE_BF16"]
else:
    cxx_args = ["-g"]
    nvcc_args = ["-O2"]


setup(
    version=get_version(),
    ext_modules=[
        CUDAExtension(
            name="points_in_rbbox.points_in_rbbox_ops",
            sources=[
                "points_in_rbbox/src/points_in_rbbox.cpp",
                "points_in_rbbox/src/points_in_rbbox_kernel.cu",
            ],
            extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
