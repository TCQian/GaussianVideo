#include "backwardvideo.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;


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
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points || radii[idx] <= 0) {
        return;
    }
    // get v_cov2d
    cov3d_to_conic_vjp(conics[idx], v_conic[idx], v_cov2d[idx]);
    
    float6 G = {
        v_cov2d[idx].x,
        v_cov2d[idx].y,
        v_cov2d[idx].z,
        v_cov2d[idx].w,
        v_cov2d[idx].u,
        v_cov2d[idx].v
    };
    
    float G_11 = G.x;
    float G_12 = G.y;
    float G_13 = G.z;
    float G_22 = G.w;
    float G_23 = G.u;
    float G_33 = G.v;

    // Corrected ordering to match forward kernel
    float l_11 = L_elements[idx].x; // l11 // l1
    float l_21 = L_elements[idx].y; // l21 // l2
    float l_31 = L_elements[idx].z; // l31 // l4
    float l_22 = L_elements[idx].w; // l22 // l3
    float l_32 = L_elements[idx].u; // l32 // l5
    float l_33 = L_elements[idx].v; // l33 // l6

    // Calculate the gradients with respect to the elements of L
    float grad_l_11 = 2 * l_11 * G_11 + 2 * G_12 * l_21 + 2 * G_13 * l_31;
    float grad_l_21 = 2 * l_11 * G_12 + 2 * l_21 * G_22 + 2 * G_23 * l_31; 
    float grad_l_31 = 2 * l_11 * G_13 + 2 * l_21 * G_23 + 2 * l_31 * G_33;
    float grad_l_22 = 2 * l_22 * G_22 + 2 * l_32 * G_23;
    float grad_l_32 = 2 * l_22 * G_23 + 2 * l_32 * G_33; 
    float grad_l_33 = 2 * l_33 * G_33;

    // Store the gradients back to the output gradient array
    // Corrected gradient storage matching the forward ordering:
    v_L_elements[idx].x = grad_l_11; // for l11
    v_L_elements[idx].y = grad_l_21; // for l21
    v_L_elements[idx].z = grad_l_31; // for l31
    v_L_elements[idx].w = grad_l_22; // for l22
    v_L_elements[idx].u = grad_l_32; // for l32
    v_L_elements[idx].v = grad_l_33; // for l33

    if (idx < 3) // Print only the first 3 for debugging
        printf("L_elements[%d]: %f, %f, %f, %f, %f, %f\n", idx, 
            L_elements[idx].x, L_elements[idx].y, L_elements[idx].z,
            L_elements[idx].w, L_elements[idx].u, L_elements[idx].v);
        printf("v_cov2d[%d]: %f, %f, %f, %f, %f, %f\n", idx, 
            v_cov2d[idx].x, v_cov2d[idx].y, v_cov2d[idx].z,
            v_cov2d[idx].w, v_cov2d[idx].u, v_cov2d[idx].v);
        printf("v_L_elements[%d]: %f, %f, %f, %f, %f, %f\n", idx, 
            v_L_elements[idx].x, v_L_elements[idx].y, v_L_elements[idx].z,
            v_L_elements[idx].w, v_L_elements[idx].u, v_L_elements[idx].v);

    v_mean2d[idx].x = v_xy[idx].x * (0.5f * img_size.x);
    v_mean2d[idx].y = v_xy[idx].y * (0.5f * img_size.y);
    v_mean2d[idx].z = v_xy[idx].z * (0.5f * img_size.z);

}
