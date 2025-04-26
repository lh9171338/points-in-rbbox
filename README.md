[<img height="23" src="https://github.com/lh9171338/Outline/blob/master/icon.jpg"/>](https://github.com/lh9171338/Outline) point-in-rbbox
===

# 功能

实现`point-in-rbbox`CUDA算子，用于计算3D旋转框内的点云，相比PyTorch版本显存有明显优化

# 依赖

- CUDA：11.8
- PyTorch：2.0.0

# 安装

- pip安装
```shell
export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX" # 支持不同GPU架构【可选】
export ENABLE_BF16=1 # 支持BF16【可选，部分GPU可能不支持】
python3 -m pip install points-in-rbbox
```

- 源码安装

```shell
git clone https://github.com/lh9171338/points-in-rbbox.git
cd points-in-rbbox

export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX" # 支持不同GPU架构【可选】
export ENABLE_BF16=1 # 支持BF16【可选，部分GPU可能不支持】
python3 setup.py install
```

# 使用

```python
from points_in_rbbox import points_in_rbbox_cuda

mask = points_in_rbbox_cuda(points, boxes)
```

# 显存&耗时

| 方法 | dtype | 显存(GB) | 耗时(s) |
| :---: | :---: | :---: | :---: |
| points_in_rbbox_torch | FP32 | 10.0 | 12 |
| points_in_rbbox_torch | FP16 | 5.3 | 10 |
| points_in_rbbox_torch | BF16 | 5.3 | 10 |
| points_in_rbbox_cuda | FP32 | 1.2 | 9 |
| points_in_rbbox_cuda | FP16 | 1.0 | 9 |
| points_in_rbbox_cuda | BF16 | 1.0 | 9 |
