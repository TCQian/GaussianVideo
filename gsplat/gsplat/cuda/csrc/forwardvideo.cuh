#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "float6.h"

__global__ void project_gaussians_video_forward_kernel(
    const int num_points,
    const float3* __restrict__ mean3d,
    const float6* __restrict__ L_elements,
    const dim3 img_size,
    const dim3 tile_bounds,
    const int timestamp,
    const float clip_thresh,
     float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit
);
