#include "forwardvideo.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include "float6.h"
#include "helpers.cuh"
#include <cstdio>

namespace cg = cooperative_groups;

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
// 这个函数应该包含的过程：计算2d cov，num_tiles_hit tile_bounds blabla 然后接入rasterize_forward？
__global__ void project_gaussians_video_forward_kernel(
    const int num_points, // 9000 (number of gaussians)
    const float3* __restrict__ means3d, // (9000, 3) (gaussian centers)
    const float6* __restrict__ L_elements, // (9000, 6) (gaussian cholesky vectors)
    const dim3 img_size, // (1920, 1080, 50) (W, H, T)
    const dim3 tile_bounds, // (120, 68, 50) (the tile max length across these W, H, T axes)
    const int timestamp, // 0 (the timestamp of the video frame, not used in this kernel)
    // Unused
    const float clip_thresh, // 0.01 (useless variable)

    // Outputs
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit
) {
    // each thread is one gaussian here
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    // Retrieve the 3D Gaussian parameters

    // This assumes means3d is clamped between [-1, 1] > validated to be true
    // This computes the center in pixel space, for example (480, 360, 25)
    float3 center = {
        0.5f * img_size.x * means3d[idx].x + 0.5f * img_size.x,
        0.5f * img_size.y * means3d[idx].y + 0.5f * img_size.y,
        0.5f * img_size.z * means3d[idx].z + 0.5f * img_size.z
    };

    // Update the L_elements indexing for the 3D covariance matrix
    float l11 = L_elements[idx].x; // scale_x // l1
    float l21 = L_elements[idx].y; // covariance_xy // l2
    float l31 = L_elements[idx].z; // covariance_xz // l4
    float l22 = L_elements[idx].w; // scale_y // l3
    float l32 = L_elements[idx].u; // covariance_yz // l5
    float l33 = L_elements[idx].v; // scale_z // l6

    // Construct the 3x3 covariance matrix
    float6 cov3d = {
        l11*l11,                            // Cxx
        l11*l21,                            // Cxy
        l11*l31,                            // Cxz
        (l21*l21 + l22*l22),                // Cyy
        (l21*l31 + l22*l32),                // Cyz
        (l31*l31 + l32*l32 + l33*l33)       // Czz
    };

    // process 3D to 2D here
    float dt = float(timestamp) - center.z;
    float cov_t = cov3d.v;
    float marginal_t = __expf(-0.5f * dt * dt / cov_t);
    bool mask = marginal_t > 0.05f;

    if (!mask) return;

    float cov_xy_11 = cov3d.x;
    float cov_xy_21 = cov3d.y;
    float cov_xy_22 = cov3d.w;

    float cov_xyt_0 = cov3d.z;
    float cov_xyt_1 = cov3d.u;

    // outer product: cov_xyt . cov_t . (cov_xyt)T
    float outer_11 = cov_xyt_0 * cov_xyt_0;
    float outer_21 = cov_xyt_0 * cov_xyt_1;
    float outer_22 = cov_xyt_1 * cov_xyt_1;

    // Compute conditional covariance: cov_xy - outer(cov_xyt, cov_xyt) / cov_t
    float cov2d_11 = cov_xy_11 - outer_11 / cov_t;
    float cov2d_21 = cov_xy_21 - outer_21 / cov_t;
    float cov2d_22 = cov_xy_22 - outer_22 / cov_t;

    // delta_mean = cov_xyt / cov_t * dt
    float delta_mean_0 = cov_xyt_0 / cov_t * dt;
    float delta_mean_1 = cov_xyt_1 / cov_t * dt;

    float2 center_xy = {
        center.x + delta_mean_0,
        center.y + delta_mean_1
    };

    float3 conic;
    float radius;

    // add a small constant to avoid zero determinant
    float3 cov2d = make_float3(
        cov2d_11 + 0.3f,
        cov2d_21,
        cov2d_22 + 0.3f
    );

    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok)
        return; // zero determinant

    conics[idx] = conic;
    xys[idx] = center_xy;
    radii[idx] = (int)radius;
    if radius <= 0 {
        // printf("%d point radius <= 0\n", idx);
        return;
    }
    uint2 tile_min, tile_max;
    get_tile_bbox(center_xy, radius, tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    // if (tile_area <= 0) {
    //     // printf("%d point bbox outside of bounds\n", idx);
    //     return;
    // }
    num_tiles_hit[idx] = tile_area;
    depths[idx] = 0.0f;
}
