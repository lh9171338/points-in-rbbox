#include <cmath>
#include <cuda_fp16.h> // 包含 __half 的定义
#ifdef ENABLE_BF16
#include <cuda_bf16.h> // 包含 __nv_bfloat16 的定义
#endif


#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))


__device__ __forceinline__ __half operator+(const __half &a, const __half &b) {
    return __hadd(a, b);
}

__device__ __forceinline__ __half operator-(const __half &a) {
    return __hneg(a);
}

__device__ __forceinline__ __half operator-(const __half &a, const __half &b) {
    return __hsub(a, b);
}

__device__ __forceinline__ __half operator*(const __half &a, const __half &b) {
    return __hmul(a, b);
}

__device__ __forceinline__ bool operator<(const __half &a, const __half &b) {
    return __hlt(a, b);
}

__device__ __forceinline__ __half fabs(const __half &a) {
    return __habs(a);
}

__device__ __forceinline__ __half cosf(const __half &a) {
    return hcos(a);
}

__device__ __forceinline__ __half sinf(const __half &a) {
    return hsin(a);
}

#ifdef ENABLE_BF16
__device__ __forceinline__ __nv_bfloat16 fabs(const __nv_bfloat16 &a) {
    return __habs(a);
}

__device__ __forceinline__ __nv_bfloat16 cosf(const __nv_bfloat16 &a) {
    return hcos(a);
}

__device__ __forceinline__ __nv_bfloat16 sinf(const __nv_bfloat16 &a) {
    return hsin(a);
}
#endif


template <typename T>
__device__ inline T float2type(float value);

template <>
__device__ inline float float2type<float>(float value) {
    return value;
}

template <>
__device__ inline __half float2type<__half>(float value) {
    return __float2half(value);
}

#ifdef ENABLE_BF16
template <>
__device__ inline __nv_bfloat16 float2type<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}
#endif

template <typename T>
__global__ void points_in_rbbox_kernel(int N, int M, const T *points, const T *boxes, bool *output)
{
    int m = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    const T* box = boxes + m * 7;
    const T* point = points + n * 3;

    // 提取box参数
    const T cx = box[0];
    const T cy = box[1];
    const T cz = box[2];
    const T half_l = box[3] * float2type<T>(0.5f);
    const T half_w = box[4] * float2type<T>(0.5f);
    const T half_h = box[5] * float2type<T>(0.5f);
    const T theta = box[6];

    // 计算旋转矩阵(绕Z轴)
    const T cos_theta = cosf(theta);
    const T sin_theta = sinf(theta);

    // 平移
    const T x1 = point[0] - cx;
    const T y1 = point[1] - cy;
    const T z = point[2] - cz;

    // 旋转
    const T x = x1 * cos_theta + y1 * sin_theta;
    const T y = -x1 * sin_theta + y1 * cos_theta;

    // 判断点是否在box内
    output[m * N + n] = (fabs(x) < half_l) && (fabs(y) < half_w) && (fabs(z) < half_h);
}

void points_in_rbbox_kernel_launcher_fp32(int N, int M, const float *points, const float *boxes, bool *output)
{
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK), M);
    dim3 threads(THREADS_PER_BLOCK);

    points_in_rbbox_kernel<<<blocks, threads, 0>>>(N, M, points, boxes, output);    
}

void points_in_rbbox_kernel_launcher_fp16(int N, int M, const __half *points, const __half *boxes, bool *output)
{
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK), M);
    dim3 threads(THREADS_PER_BLOCK);

    points_in_rbbox_kernel<<<blocks, threads, 0>>>(N, M, points, boxes, output);
}

#ifdef ENABLE_BF16
void points_in_rbbox_kernel_launcher_bf16(int N, int M, const __nv_bfloat16 *points, const __nv_bfloat16 *boxes, bool *output)
{
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK), M);
    dim3 threads(THREADS_PER_BLOCK);

    points_in_rbbox_kernel<<<blocks, threads, 0>>>(N, M, points, boxes, output);
}
#endif
