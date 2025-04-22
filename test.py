# -*- encoding: utf-8 -*-
"""
@File    :   test.py
@Time    :   2025/04/21 13:10:38
@Author  :   lh9171338
@Version :   1.0
@Contact :   2909171338@qq.com
"""

import unittest
import pickle
import torch
import logging
import numpy as np
from lh_tool.time_consumption import TimeConsumption
from lh_tool.monitor_gpu import GPUPeakMemoryMonitor
from points_in_rbbox import (
    points_in_rbbox_numpy,
    points_in_rbbox_torch,
    points_in_rbbox_cuda,
)


gpu_format_func = lambda x: f"{x / (1024 ** 3):.1f}GB"
time_format_func = lambda x: f"{x:.0f}s"


class Test(unittest.TestCase):
    """Test"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_file = "./data.pkl"
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        
        self.small_points = data["points"][:, :3]
        self.points = data["points"][:, :3].repeat(40, axis=0)
        self.boxes = data["boxes"]
        logging.info(f"points shape: {self.points.shape}")
        logging.info(f"boxes shape: {self.boxes.shape}")

        self.run_times = 20

    def test_precision(self):
        logging.info("Test precision")
        mask_numpy = points_in_rbbox_numpy(self.small_points, self.boxes)
        mask_torch_fp32 = points_in_rbbox_torch(self.small_points, self.boxes, dtype=torch.float32)
        mask_torch_fp16 = points_in_rbbox_torch(self.small_points, self.boxes, dtype=torch.float16)
        mask_torch_bf16 = points_in_rbbox_torch(self.small_points, self.boxes, dtype=torch.bfloat16)
        mask_cuda_fp32 = points_in_rbbox_cuda(self.small_points, self.boxes, dtype=torch.float32)
        mask_cuda_fp16 = points_in_rbbox_cuda(self.small_points, self.boxes, dtype=torch.float16)
        try:
            mask_cuda_bf16 = points_in_rbbox_cuda(self.small_points, self.boxes, dtype=torch.bfloat16)
        except RuntimeError:
            logging.warning("CUDA BF16 is not supported")
            mask_cuda_bf16 = None
        self.assertTrue((mask_numpy == mask_torch_fp32).all())
        self.assertTrue((mask_numpy == mask_torch_fp16).all())
        self.assertTrue((mask_numpy == mask_torch_bf16).all())
        self.assertTrue((mask_numpy == mask_cuda_fp32).all())
        self.assertTrue((mask_numpy == mask_cuda_fp16).all())
        if mask_cuda_bf16 is not None:
            self.assertTrue((mask_numpy == mask_cuda_bf16).all())

    def test_precision_v2(self):
        logging.info("Test precision V2")
        indices_list_numpy = points_in_rbbox_numpy(self.small_points, self.boxes, return_indices=True)
        indices_list_torch_fp32 = points_in_rbbox_torch(self.small_points, self.boxes, dtype=torch.float32, return_indices=True)
        indices_list_torch_fp16 = points_in_rbbox_torch(self.small_points, self.boxes, dtype=torch.float16, return_indices=True)
        indices_list_torch_bf16 = points_in_rbbox_torch(self.small_points, self.boxes, dtype=torch.bfloat16, return_indices=True)
        indices_list_cuda_fp32 = points_in_rbbox_cuda(self.small_points, self.boxes, dtype=torch.float32, return_indices=True)
        indices_list_cuda_fp16 = points_in_rbbox_cuda(self.small_points, self.boxes, dtype=torch.float16, return_indices=True)
        try:
            indices_list_cuda_bf16 = points_in_rbbox_cuda(self.small_points, self.boxes, dtype=torch.bfloat16, return_indices=True)
        except RuntimeError:
            logging.warning("CUDA BF16 is not supported")
            indices_list_cuda_bf16 = None

        def assert_equal(a, b):
            a = np.concatenate(a)
            b = np.concatenate(b)
            self.assertTrue(a.shape == b.shape)
            self.assertTrue((a == b).all())

        assert_equal(indices_list_numpy, indices_list_torch_fp32)
        assert_equal(indices_list_numpy, indices_list_torch_fp16)
        assert_equal(indices_list_numpy, indices_list_torch_bf16)
        assert_equal(indices_list_numpy, indices_list_cuda_fp32)
        assert_equal(indices_list_numpy, indices_list_cuda_fp16)
        if indices_list_cuda_bf16 is not None:
            assert_equal(indices_list_numpy, indices_list_cuda_bf16)

    @TimeConsumption(format_func=time_format_func)
    @GPUPeakMemoryMonitor(format_func=gpu_format_func)
    def test_torch_fp32(self):
        for _ in range(self.run_times):
            points_in_rbbox_torch(self.points, self.boxes, dtype=torch.float32)

        self.assertTrue(True)

    @TimeConsumption(format_func=time_format_func)
    @GPUPeakMemoryMonitor(format_func=gpu_format_func)
    def test_torch_fp16(self):
        for _ in range(self.run_times):
            points_in_rbbox_torch(self.points, self.boxes, dtype=torch.float16)

        self.assertTrue(True)

    @TimeConsumption(format_func=time_format_func)
    @GPUPeakMemoryMonitor(format_func=gpu_format_func)
    def test_torch_bf16(self):
        for _ in range(self.run_times):
            points_in_rbbox_torch(self.points, self.boxes, dtype=torch.bfloat16)

        self.assertTrue(True)

    @TimeConsumption(format_func=time_format_func)
    @GPUPeakMemoryMonitor(format_func=gpu_format_func)
    def test_cuda_fp32(self):
        for _ in range(self.run_times):
            points_in_rbbox_cuda(self.points, self.boxes, dtype=torch.float32)

        self.assertTrue(True)

    @TimeConsumption(format_func=time_format_func)
    @GPUPeakMemoryMonitor(format_func=gpu_format_func)
    def test_cuda_fp16(self):
        for _ in range(self.run_times):
            points_in_rbbox_cuda(self.points, self.boxes, dtype=torch.float16)

        self.assertTrue(True)

    @TimeConsumption(format_func=time_format_func)
    @GPUPeakMemoryMonitor(format_func=gpu_format_func)
    def test_cuda_bf16(self):
        for _ in range(self.run_times):
            points_in_rbbox_cuda(self.points, self.boxes, dtype=torch.bfloat16)

        self.assertTrue(True)


if __name__ == "__main__":
    # set base logging config
    fmt = "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    unittest.main()
