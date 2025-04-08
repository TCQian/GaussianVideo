#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "helpers.cuh"

__global__ void project_gaussians_video_backward_kernel(
    const int num_points,
    const float3* __restrict__ means2d,
    const float6* __restrict__ L_elements,
    const dim3 img_size,
    const int* __restrict__ radii,
    const float6* __restrict__ conics,
    const float3* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float6* __restrict__ v_conic,
    float6* __restrict__ v_cov2d,
    float3* __restrict__ v_mean2d,
    float6* __restrict__ v_L_elements
);
