#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // 包含 __half 的定义
#ifdef ENABLE_BF16
#include <cuda_bf16.h> // 包含 __nv_bfloat16 的定义
#endif

void points_in_rbbox_kernel_launcher_fp32(int N, int M, const float *points, const float *boxes, bool *output);
void points_in_rbbox_kernel_launcher_fp16(int N, int M, const __half *points, const __half *boxes, bool *output);
#ifdef ENABLE_BF16
void points_in_rbbox_kernel_launcher_bf16(int N, int M, const __nv_bfloat16 *points, const __nv_bfloat16 *boxes, bool *output);
#endif

void points_in_rbbox_wrapper(at::Tensor points_tensor, at::Tensor boxes_tensor, at::Tensor output_tensor)
{
    int device_id = points_tensor.device().index();
    cudaSetDevice(device_id);
    int N = points_tensor.size(0);
    int M = boxes_tensor.size(0);
    bool *output = output_tensor.data_ptr<bool>();
    if (points_tensor.scalar_type() == at::kHalf) {
        using T = __half;
        const T *points = static_cast<T*>(points_tensor.data_ptr());
        const T *boxes = static_cast<T*>(boxes_tensor.data_ptr());
        points_in_rbbox_kernel_launcher_fp16(N, M, points, boxes, output);
#ifdef ENABLE_BF16
    } else if (points_tensor.scalar_type() == at::kBFloat16) {
        using T = __nv_bfloat16;
        const T *points = static_cast<T*>(points_tensor.data_ptr());
        const T *boxes = static_cast<T*>(boxes_tensor.data_ptr());
        points_in_rbbox_kernel_launcher_bf16(N, M, points, boxes, output);
#endif
    } else {
        using T = float;
        const T *points = static_cast<T*>(points_tensor.data_ptr());
        const T *boxes = static_cast<T*>(boxes_tensor.data_ptr());
        points_in_rbbox_kernel_launcher_fp32(N, M, points, boxes, output);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("points_in_rbbox_wrapper", &points_in_rbbox_wrapper, "points_in_rbbox_wrapper");
}
