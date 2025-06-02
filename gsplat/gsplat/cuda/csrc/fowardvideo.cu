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
    const float3* __restrict__ means2d, // (9000, 3) (gaussian centers)
    const float6* __restrict__ L_elements, // (9000, 6) (gaussian cholesky vectors)
    const dim3 img_size, // (1920, 1080, 50) (W, H, T)
    const dim3 tile_bounds, // (120, 68, 50) (the tile max length across these W, H, T axes)
    
    // Unused
    const float clip_thresh, // 0.01 (useless variable)

    // Outputs
    float3* __restrict__ xys,  // pixel center, e.g. (480, 360, 25)
    float* __restrict__ depths, // 0.0f Unused
    int* __restrict__ radii, // pixel radius, e.g. 16
    float6* __restrict__ conics, // gaussian conics representing shape of gaussian
    int32_t* __restrict__ num_tiles_hit // total number of tiles hit e.g. 500
) {
    // each thread is one gaussian here
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    // Retrieve the 3D Gaussian parameters

    // This assumes means2d is clamped between [-1, 1] > validated to be true
    // This computes the center in pixel space, for example (480, 360, 25)
    float3 center = {
        0.5f * img_size.x * means2d[idx].x + 0.5f * img_size.x,
        0.5f * img_size.y * means2d[idx].y + 0.5f * img_size.y,
        0.5f * img_size.z * means2d[idx].z + 0.5f * img_size.z
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
        l11*l11*1.5,                            // Cxx
        l11*l21,                            // Cxy
        l11*l31,                            // Cxz
        (l21*l21 + l22*l22)*1.5,                // Cyy
        (l21*l31 + l22*l32),                // Cyz
        (l31*l31 + l32*l32 + l33*l33)*1.5       // Czz
    };
    
    float6 conic;
    float radius;

    // Computes the conic and radius of the gaussian based on the covariance matrix
    // radius is in pixel space
    bool ok = compute_cov3d_bounds(cov3d, conic, radius);
    if (!ok) {
        printf("Gaussian %d with L_elements (%.2f, %.2f, %.2f, %.2f, %.2f, %.2f) has zero determinant.\n",
               idx, L_elements[idx].x, L_elements[idx].y, L_elements[idx].z,
               L_elements[idx].w, L_elements[idx].u, L_elements[idx].v);
        return; // zero determinant
    }

    // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
    conics[idx] = conic;
    xys[idx] = center;
    radii[idx] = (int)radius;
    uint3 tile_min, tile_max;

    // Computes a bounding box of the gaussian based on the center, radius, tile_bounds
    // the bounding box consists of tile_min, and tile_max
    // tile_min: minimum x, y, z tiles which intersect with gaussian
    // tile_max: maximum x, y, z tiles which intersect with gaussian
    get_tile_3d_bbox(center, radius, tile_bounds, tile_min, tile_max);
    
    // Compute the volume of the whole intersection space
    int32_t tile_volume = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y) * (tile_max.z - tile_min.z);
    
    if (tile_volume <= 0) {
        return; // Check volume instead of area
    }

    // record as number of tiles hit
    num_tiles_hit[idx] = tile_volume; // Update to track 3D tiles

    // Useless value
    depths[idx] = 0.0f;

    // printf("[DEBUG] Gaussian %d: center=(%.2f, %.2f, %.2f), radius=%d, conic=(%.2f, %.2f, %.2f, %.2f, %.2f, %.2f), num_tiles_hit=%d\n",
    //    idx, center.x, center.y, center.z, radii[idx], 
    //    conics[idx].x, conics[idx].y, conics[idx].z, 
    //    conics[idx].w, conics[idx].u, conics[idx].v, num_tiles_hit[idx]);
}
