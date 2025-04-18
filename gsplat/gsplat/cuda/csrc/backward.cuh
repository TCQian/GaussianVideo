#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "float6.h"

// for f : R(n) -> R(m), J in R(m, n),
// v is cotangent in R(m), e.g. dL/df in R(m),
// compute vjp i.e. vT J -> R(n)
__global__ void project_gaussians_backward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* __restrict__ viewmat,
    const float* __restrict__ projmat,
    const float4 intrins,
    const dim3 img_size,
    const float* __restrict__ cov3d,
    const int* __restrict__ radii,
    const float3* __restrict__ conics,
    const float2* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float3* __restrict__ v_conic,
    float3* __restrict__ v_cov2d,
    float* __restrict__ v_cov3d,
    float3* __restrict__ v_mean3d,
    float3* __restrict__ v_scale,
    float4* __restrict__ v_quat
);

// compute jacobians of output image wrt binned and sorted gaussians
__global__ void nd_rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float* __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb,
    float* __restrict__ v_opacity,
    float* __restrict__ workspace
);

__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity
);

__global__ void rasterize_backward_sum_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity
);

__global__ void rasterize_backward_sum_general_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float* __restrict__ betas,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity,
    float* __restrict__ v_beta
);

__global__ void nd_rasterize_backward_sum_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float* __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb,
    float* __restrict__ v_opacity,
    float* __restrict__ workspace
);

__device__ void project_cov3d_ewa_vjp(
    const float3 &mean3d,
    const float *cov3d,
    const float *viewmat,
    const float fx,
    const float fy,
    const float3 &v_cov2d,
    float3 &v_mean3d,
    float *v_cov3d
);

__device__ void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float *v_cov3d,
    float3 &v_scale,
    float4 &v_quat
);

__global__ void rasterize_video_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const float time,
    const float vis_thresold,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float* __restrict__ means_t,
    const float* __restrict__ lambda,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity,
    float* __restrict__ v_means_t,
    float* __restrict__ v_lambda
);

__global__ void rasterize_backward_sum_kernel_video(
    const dim3 tile_bounds,              // 3D tile grid dimensions
    const dim3 img_size,                 // 3D image dimensions (width, height, depth)
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,  // 3D Gaussian ranges for each tile
    const float3* __restrict__ xyzs,     // 3D positions (x, y, z) for each Gaussian
    const float6* __restrict__ conics,   // Shape parameters for each Gaussian
    const float3* __restrict__ rgbs,     // Colors (R, G, B) for each Gaussian
    const float* __restrict__ opacities, // Opacity for each Gaussian
    const float3& __restrict__ background, // Background color
    const float* __restrict__ final_Ts,  // Final transmittance for each voxel
    const int* __restrict__ final_index, // Index of the last Gaussian for each voxel
    const float3* __restrict__ v_output, // Voxel gradients for the 3D image colors
    const float* __restrict__ v_output_alpha, // Voxel gradients for alpha values
    float3* __restrict__ v_xy,           // Gradient wrt (x, y, z) for each Gaussian
    float6* __restrict__ v_conic,        // Gradient wrt conic parameters for each Gaussian
    float3* __restrict__ v_rgb,          // Gradient wrt color for each Gaussian
    float* __restrict__ v_opacity        // Gradient wrt opacity for each Gaussian
);