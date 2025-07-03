#include "backwardvideo.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;


__global__ void project_gaussians_video_backward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float6* __restrict__ L_elements,
    const dim3 img_size,
    const int timestamp,
    const int* __restrict__ radii,
    const float3* __restrict__ conics,
    const float2* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float3* __restrict__ v_conic,
    float6* __restrict__ v_cov3d,
    float3* __restrict__ v_mean3d,
    float6* __restrict__ v_L_elements
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points || radii[idx] <= 0) {
        return;
    }
    // get v_cov2d
    float3 v_cov2d;
    cov2d_to_conic_vjp(conics[idx], v_conic[idx], &v_cov2d);
    float G_11 = v_cov2d.x; // dL/dcov2d_11
    float G_12 = v_cov2d.y; // dL/dcov2d_12, which is the same as dL/dcov2d_21
    float G_22 = v_cov2d.z; // dL/dcov2d_22

    float3 center = {
        0.5f * img_size.x * means2d[idx].x + 0.5f * img_size.x,
        0.5f * img_size.y * means2d[idx].y + 0.5f * img_size.y,
        0.5f * img_size.z * means2d[idx].z + 0.5f * img_size.z
    };

    // Corrected ordering to match forward kernel
    float l_11 = L_elements[idx].x; // l11 // l1
    float l_21 = L_elements[idx].y; // l21 // l2
    float l_31 = L_elements[idx].z; // l31 // l4
    float l_22 = L_elements[idx].w; // l22 // l3
    float l_32 = L_elements[idx].u; // l32 // l5
    float l_33 = L_elements[idx].v; // l33 // l6

    / Construct the 3x3 covariance matrix
    float6 cov3d = {
        l11*l11,                            // Cxx
        l11*l21,                            // Cxy
        l11*l31,                            // Cxz
        (l21*l21 + l22*l22),                // Cyy
        (l21*l31 + l22*l32),                // Cyz
        (l31*l31 + l32*l32 + l33*l33)       // Czz
    };

    float cov_xyt_0 = cov3d.z;
    float cov_xyt_1 = cov3d.u;

    float dt = float(timestamp) - center.z;
    float cov_t = cov3d.v;
    float marginal_t = __expf(-0.5f * dt * dt / cov_t);
    bool mask = marginal_t > 0.05f;

    if (!mask) return;

    // cov2d = cov_xy - outer(cov_xyt, cov_xyt) / cov_t
    // Compute dL/dcov_xyt = dL/dcov2d * dcov2d/dcov_xyt
    // dcov2d/dcov_xyt = d(-1/cov_t * cov_xyt * cov_xyt^T)/dcov_xyt
    // dL/dcov_xyt =  -1/cov_t * d(v_cov2d * cov_xyt * cov_xyt^T)/dcov_xyt
    // v_cov2d * cov_xyt * cov_xyt^T  = cov_xyt^T * v_cov2d * cov_xyt // trace cyclic
    // d(cov_xyt^T * v_cov2d * cov_xyt)/dcov_xyt =  2 * v_cov2d * cov_xyt //quadratic form
    // Thus, dL/dcov_xyt =  -1/cov_t * 2 * v_cov2d * cov_xyt
    float3 v_cov_xyt = {
        (G_11 * cov_xyt_0 + G_12 * cov_xyt_1 * 0.5) * -2.0f / cov_t,
        (G_12 * cov_xyt_0 * 0.5 + G_22 * cov_xyt_1) * -2.0f / cov_t,
    };

    // Compute dL/dcov_t = dL/dcov2d * dcov2d/dcov_t
    // dcov2d/dcov_t = (cov_xyt * cov_xyt^T) / (cov_t * cov_t)
    // dL/dcov_t = (v_cov2d * cov_xyt * cov_xyt^T) / (cov_t * cov_t)
    //           = (cov_xyt^T * v_cov2d * cov_xyt) / (cov_t * cov_t) 
    float v_cov_t = (
        cov_xyt_0 * G11 * cov_xyt_0 +
        2 * cov_xyt_0 * G12 * cov_xyt_1 +
        cov_xyt_1 * G22 * cov_xyt_1
    ) / (cov_t * cov_t);

    // mean2d = center_xy + (cov_xyt / cov_t * dt)
    // dL/dcov_xyt = dL/dmean2d * dmean2d/dcov_xyt
    //             = dL/dmean2d * (1/cov_t * dt)
    v_cov_xyt.x += v_xy[idx].x * (dt / cov_t);
    v_cov_xyt.y += v_xy[idx].y * (dt / cov_t);

    // dL/dcov_t = dL/dmean2d * dmean2d/dcov_t
    //           = dL/dmean2d * (-1 * cov_xyt * dt / cov_t^2)
    v_cov_t += (v_xy[idx].x * cov_xyt_0 + v_xy[idx].y * cov_xyt_1) * (-dt / (cov_t * cov_t));

    // dL/dmean_t = dL/dmean2d * dmean2d/dmean_t
    //            = dL/dmean2d * (-1 * cov_xyt / cov_t)
    float v_mean_t = - (v_xy[idx].x * cov_xyt_0 + v_xy[idx].y * cov_xyt_1) / cov_t;

    // gradient of the 3D covariance matrix
    float6 v_cov3d[idx] = {
        G_11, // dL/dcov2d_11
        G_12, // dL/dcov2d_12
        v_cov_xyt.x, // dL/dcov2d_13
        G_22, // dL/dcov2d_22
        v_cov_xyt.y, // dL/dcov2d_23
        v_cov_t  // dL/dcov2d_33
    };

    float G_11 = v_cov3d[idx].x;
    float G_12 = v_cov3d[idx].y;
    float G_13 = v_cov3d[idx].z;
    float G_22 = v_cov3d[idx].w;
    float G_23 = v_cov3d[idx].u;
    float G_33 = v_cov3d[idx].v;

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

    v_mean3d[idx].x = v_xy[idx].x * (0.5f * img_size.x);
    v_mean3d[idx].y = v_xy[idx].y * (0.5f * img_size.y);
    v_mean3d[idx].z = v_mean_t * (0.5f * img_size.z);

}
