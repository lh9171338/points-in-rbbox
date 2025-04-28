# -*- encoding: utf-8 -*-
"""
@File    :   setup.py
@Time    :   2025/04/18 16:30:05
@Author  :   lh9171338
@Version :   1.0
@Contact :   2909171338@qq.com
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


if os.getenv("ENABLE_BF16", "0") == "1":
    print("BF16 is enabled")
    cxx_args = ["-g", "-DENABLE_BF16"]
    nvcc_args = ["-O2", "-DENABLE_BF16"]
else:
    cxx_args = ["-g"]
    nvcc_args = ["-O2"]

setup(
    name="points_in_rbbox",
    version="1.1.6",
    author="lh9171338",
    author_email="lihao2015@whu.edu.cn",
    description="point-in-rbbox CUDA ops",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/lh9171338/points-in-rbbox",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["points_in_rbbox"],
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
