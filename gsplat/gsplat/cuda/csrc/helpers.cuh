#pragma once
#include "config.h"
#include <cuda_runtime.h>
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include <iostream>
#include "float6.h"

inline __device__ float ndc2pix(const float x, const float W, const float cx) {
    return 0.5f * W * x + cx - 0.5f;
}

inline __device__ void get_bbox(
    const float2 center,
    const float2 dims,
    const dim3 img_size,
    uint2 &bb_min,
    uint2 &bb_max
) {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

inline __device__ void get_tile_bbox(
    const float2 pix_center,
    const float pix_radius,
    const dim3 tile_bounds,
    uint2 &tile_min,
    uint2 &tile_max
) {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    float2 tile_center = {
        pix_center.x / (float)BLOCK_X, pix_center.y / (float)BLOCK_Y
    };
    float2 tile_radius = {
        pix_radius / (float)BLOCK_X, pix_radius / (float)BLOCK_Y
    };
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

inline __device__ bool
compute_cov2d_bounds(const float3 cov2d, float3 &conic, float &radius) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    if (fabsf(det) < 1e-12f) {
        return false;  // singular or nearly singular
    }
    float inv_det = 1.f / det;

    // inverse of 2x2 cov2d matrix
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det));
    float v2 = b - sqrt(max(0.1f, b * b - det));
    // take 3 sigma of covariance
    radius = ceil(3.f * sqrt(max(v1, v2)));
    return true;
}

// compute vjp from df/d_conic to df/c_cov2d
inline __device__ void cov2d_to_conic_vjp(
    const float3 &conic, const float3 &v_conic, float3 &v_cov2d
) {
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    glm::mat2 X = glm::mat2(conic.x, conic.y, conic.y, conic.z);
    glm::mat2 G = glm::mat2(v_conic.x, v_conic.y, v_conic.y, v_conic.z);
    glm::mat2 v_Sigma = -X * G * X;
    v_cov2d.x = v_Sigma[0][0];
    v_cov2d.y = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d.z = v_Sigma[1][1];
}

// helper for applying R * p + T, expect mat to be ROW MAJOR
inline __device__ float3 transform_4x3(const float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
    };
    return out;
}

// helper to apply 4x4 transform to 3d vector, return homo coords
// expects mat to be ROW MAJOR
inline __device__ float4 transform_4x4(const float *mat, const float3 p) {
    float4 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
        mat[12] * p.x + mat[13] * p.y + mat[14] * p.z + mat[15],
    };
    return out;
}

inline __device__ float2 project_pix(
    const float *mat, const float3 p, const dim3 img_size, const float2 pp
) {
    // ROW MAJOR mat
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);
    float3 p_proj = {p_hom.x * rw, p_hom.y * rw, p_hom.z * rw};
    return {
        ndc2pix(p_proj.x, img_size.x, pp.x), ndc2pix(p_proj.y, img_size.y, pp.y)
    };
}

// given v_xy_pix, get v_xyz
inline __device__ float3 project_pix_vjp(
    const float *mat, const float3 p, const dim3 img_size, const float2 v_xy
) {
    // ROW MAJOR mat
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);

    float3 v_ndc = {0.5f * img_size.x * v_xy.x, 0.5f * img_size.y * v_xy.y};
    float4 v_proj = {
        v_ndc.x * rw, v_ndc.y * rw, 0., -(v_ndc.x + v_ndc.y) * rw * rw
    };
    // df / d_world = df / d_cam * d_cam / d_world
    // = v_proj * P[:3, :3]
    return {
        mat[0] * v_proj.x + mat[4] * v_proj.y + mat[8] * v_proj.z,
        mat[1] * v_proj.x + mat[5] * v_proj.y + mat[9] * v_proj.z,
        mat[2] * v_proj.x + mat[6] * v_proj.y + mat[10] * v_proj.z
    };
}

inline __device__ glm::mat3 quat_to_rotmat(const float4 quat) {
    // quat to rotation matrix
    float s = rsqrtf(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    // glm matrices are column-major
    return glm::mat3(
        1.f - 2.f * (y * y + z * z),
        2.f * (x * y + w * z),
        2.f * (x * z - w * y),
        2.f * (x * y - w * z),
        1.f - 2.f * (x * x + z * z),
        2.f * (y * z + w * x),
        2.f * (x * z + w * y),
        2.f * (y * z - w * x),
        1.f - 2.f * (x * x + y * y)
    );
}

// inline __device__ glm::mat3 rotor_to_rotmat(const float4 rot) {
//     // quat to rotation matrix
//     float s = rsqrtf(
//         rot.x * rot.x + rot.y * rot.y + rot.z * rot.z + rot.w * rot.w
//     );
//     float x = rot.x * s;
//     float y = rot.y * s;
//     float z = rot.z * s;
//     float w = rot.w * s;

//     // glm matrices are column-major
//     return glm::mat3(
//         x * x - y * y - z * z + w * w,
//         -2.f * (x * y + w * z),
//         2.f * (y * w - x * z),
//         2.f * (x * y - w * z),
//         x * x - y * y + z * z - w * w,
//         -2.f * (y * z + w * x),
//         2.f * (y * w + x * z),
//         2.f * (x * w - y * z),
//         x * x + y * y - z * z - w * w
//     );
// }



inline __device__ float4
quat_to_rotmat_vjp(const float4 quat, const glm::mat3 v_R) {
    float s = rsqrtf(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    float4 v_quat;
    // v_R is COLUMN MAJOR
    // w element stored in x field
    v_quat.x =
        2.f * (
                  // v_quat.w = 2.f * (
                  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                  z * (v_R[0][1] - v_R[1][0])
              );
    // x element in y field
    v_quat.y =
        2.f *
        (
            // v_quat.x = 2.f * (
            -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
            z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
        );
    // y element in z field
    v_quat.z =
        2.f *
        (
            // v_quat.y = 2.f * (
            x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
            z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
        );
    // z element in w field
    v_quat.w =
        2.f *
        (
            // v_quat.z = 2.f * (
            x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
            2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
        );
    return v_quat;
}

inline __device__ glm::mat3
scale_to_mat(const float3 scale, const float glob_scale) {
    glm::mat3 S = glm::mat3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

// inline __device__ glm::mat3
// inverse_scale_to_mat(const float3 scale, const float glob_scale) {
//     glm::mat3 S = glm::mat3(1.f);
//     S[0][0] = 1 / (glob_scale * scale.x);
//     S[1][1] = 1 / (glob_scale * scale.y);
//     S[2][2] = 1 / (glob_scale * scale.z);
//     return S;
// }

inline __device__ glm::mat3
triangular_mat(const float3 diag_elements, const float3 non_diag_elements) {
    glm::mat3 L = glm::mat3(1.f);
    L[0][0] = diag_elements.x;
    L[1][1] = diag_elements.y;
    L[2][2] = diag_elements.z;
    L[1][0] = non_diag_elements.x;
    L[2][0] = non_diag_elements.y;
    L[2][1] = non_diag_elements.z;
    return L;
}


inline __device__ glm::mat2
scale_to_mat2d(const float2 scale) {
    glm::mat2 S = glm::mat2(1.f);
    S[0][0] = scale.x;
    S[1][1] = scale.y;
    return S;
}

inline __device__ glm::mat2 rotmat2d(const float rot) {
    // quat to rotation matrix
    float cosr = cos(rot);
    float sinr = sin(rot);

    glm::mat2 R = glm::mat2(cosr);
    R[0][1] = -sinr;
    R[1][0] = sinr;

    // glm matrices are column-major
    return R;
}

inline __device__ glm::mat2 rotmat2d_gradient(const float rot) {
    // quat to rotation matrix
    float cosr = cos(rot);
    float sinr = sin(rot);

    glm::mat2 R = glm::mat2(-sinr);
    R[0][1] = -cosr;
    R[1][0] = cosr;

    // glm matrices are column-major
    return R;
}

// device helper for culling near points
inline __device__ bool clip_near_plane(
    const float3 p, const float *viewmat, float3 &p_view, float thresh
) {
    p_view = transform_4x3(viewmat, p);
    if (p_view.z <= thresh) {
        return true;
    }
    return false;
}

// GAUSSIAN VIDEO

inline __device__ bool
compute_cov3d_bounds(const float6 cov3d, float6 &conic, float &radius)
{
    /************************************************
     * 1) Extract matrix elements for clarity
     ************************************************/
    float Cxx = cov3d.x;  // m00
    float Cxy = cov3d.y;  // m01 = m10
    float Cxz = cov3d.z;  // m02 = m20
    float Cyy = cov3d.w;  // m11
    float Cyz = cov3d.u;  // m12 = m21
    float Czz = cov3d.v;  // m22

    /************************************************
     * 2) Compute determinant using the full 3x3
     ************************************************/
    // det(M) = a*determinant(submatrix) - b*... etc.
    // For symmetric 3x3:
    //   m00*(m11*m22 - m12*m12)
    // - m01*(m01*m22 - m02*m12)
    // + m02*(m01*m12 - m02*m11)

    float det = Cxx * (Cyy*Czz - Cyz*Cyz)
              - Cxy * (Cxy*Czz - Cxz*Cyz)
              + Cxz * (Cxy*Cyz - Cxz*Cyy);

    if (fabsf(det) < 1e-12f) {
        return false;  // singular or nearly singular
    }

    /************************************************
     * 3) Compute the matrix inverse via cofactors
     ************************************************/
    float invDet = 1.f / det;

    // Cofactor matrix (transpose of the minor matrix).
    // a00, a01, a02, a10, a11, etc. Then we multiply by invDet.

    // minors for first row
    float A00 = (Cyy*Czz - Cyz*Cyz);
    float A01 = -(Cxy*Czz - Cxz*Cyz);
    float A02 = (Cxy*Cyz - Cxz*Cyy);

    // minors for second row
    // (notice symmetrical patterns)
    float A10 = A01;  // since it's symmetrical
    float A11 = (Cxx*Czz - Cxz*Cxz);
    float A12 = -(Cxx*Cyz - Cxy*Cxz);

    // minors for third row
    float A20 = A02;
    float A21 = A12; // symmetrical
    float A22 = (Cxx*Cyy - Cxy*Cxy);

    // Now store them in the same float6 layout: [invCxx, invCxy, invCxz, invCyy, invCyz, invCzz]
    // M^-1 = (1/det) * cofactorMatrix^T
    conic.x = A00 * invDet;  // invCxx
    conic.y = A01 * invDet;  // invCxy
    conic.z = A02 * invDet;  // invCxz
    conic.w = A11 * invDet;  // invCyy
    conic.u = A12 * invDet;  // invCyz
    conic.v = A22 * invDet;  // invCzz

    /************************************************
     * 4) Approximate the largest eigenvalue
     *    via power iteration
     ************************************************/
    // We want: max eigenvalue of original Cov (not the inverse).
    // We'll apply Cov * vector repeatedly to find the principal eigenvector.

    // Build "Cov" as full 3x3 for multiplication:
    // We'll do a small vector multiply in a loop.
    auto matvec = [&](const float3 &v) {
        return make_float3(
            Cxx*v.x + Cxy*v.y + Cxz*v.z,
            Cxy*v.x + Cyy*v.y + Cyz*v.z,
            Cxz*v.x + Cyz*v.y + Czz*v.z
        );
    };

    float3 v = make_float3(1.f, 1.f, 1.f);  // initial guess
    for (int i = 0; i < 5; i++) {
        float3 w = matvec(v);
        float len = sqrtf(w.x*w.x + w.y*w.y + w.z*w.z);
        if (len < 1e-12f) break;
        w.x /= len; w.y /= len; w.z /= len;
        v = w;
    }
    // Approx eigenvalue = v^T * Cov * v
    float3 Mv = matvec(v);
    float largestEigenVal = (v.x*Mv.x + v.y*Mv.y + v.z*Mv.z);

    if (largestEigenVal < 0.f) {
        // numerical issues -> clamp
        largestEigenVal = 0.f;
    }

    /************************************************
     * 5) The bounding radius = 3*sqrt(largestEigenVal)
     ************************************************/
    radius = ceilf(3.f * sqrtf(largestEigenVal));

    return true;
}


inline __device__ void get_3d_bbox(
    const float3 center, // tile_center, e.g. (30, 22.5, 25)
    const float3 dims, // radius, e.g. (2, 2, 32)
    const dim3 img_size, // tile_bonds e.g. (120, 68, 50)

    // Outputs
    uint3 &bb_min, // minimum x, y, z tiles which intersect with gaussian
    uint3 &bb_max // maximum x, y, z tiles which intersect with gaussian
) {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
    bb_min.z = min(max(0, (int)(center.z - dims.z)), img_size.z); // New line for z-dimension
    bb_max.z = min(max(0, (int)(center.z + dims.z + 1)), img_size.z); // New line for z-dimension
}

inline __device__ void get_tile_3d_bbox(
    const float3 pix_center, // pixel center, for e.g. (480, 360, 25)
    const float pix_radius, // radius value, for e.g. 32
    const dim3 tile_bounds, // (120, 68, 50) (the tile max length across these W, H, T axes)
    
    // Outputs
    uint3 &tile_min, // Minimum tile
    uint3 &tile_max // Maximum tile
) {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)

    // transforms pixel center to tile center
    // for e.g. transform (480, 360, 25) to (30, 22.5, 25) in tile space
    float3 tile_center = {
        pix_center.x / (float)BLOCK_X,
        pix_center.y / (float)BLOCK_Y,
        pix_center.z / (float)BLOCK_Z  // New line for z-dimension
    };

    // In this case, we are assuming a perfect ellipsoid of some form?
    // Assuming radius = 32, that means
    // tile_radius = { 2, 2, 32 }
    float3 tile_radius = {
        pix_radius / (float)BLOCK_X,
        pix_radius / (float)BLOCK_Y,
        pix_radius / (float)BLOCK_Z  // New line for z-dimension
    };
    get_3d_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

// compute vjp from df/d_conic to df/d_cov3d
inline __device__ void cov3d_to_conic_vjp(
    const float6 &conic, const float6 &v_conic, float6 &v_cov3d
) {
    // conic = inverse cov3d
    // df/d_cov3d = -conic * df/d_conic * conic

    // Create a 3x3 matrix from the conic representation
    glm::mat3 X = glm::mat3(
        conic.x, conic.y, conic.z,   // first row
        conic.y, conic.w, conic.u,   // second row
        conic.z, conic.u, conic.v    // third row
    );

    // Create a gradient matrix from v_conic
    glm::mat3 G = glm::mat3(
        v_conic.x, v_conic.y, v_conic.z,   // first row
        v_conic.y, v_conic.w, v_conic.u,   // second row
        v_conic.z, v_conic.u, v_conic.v    // third row
    );

    // Compute the gradient with respect to the covariance matrix
    glm::mat3 v_Sigma = -X * G * X;

    // Store the results in v_cov3d
    v_cov3d.x = v_Sigma[0][0];// * 1.5;                      // d/dCxx
    v_cov3d.y = v_Sigma[1][0] + v_Sigma[0][1];        // d/dCxy
    v_cov3d.z = v_Sigma[2][0] + v_Sigma[0][2];        // d/dCxz
    v_cov3d.w = v_Sigma[1][1];// * 1.5;                        // d/dCyy
    v_cov3d.u = v_Sigma[2][1] + v_Sigma[1][2];        // d/dCyz
    v_cov3d.v = v_Sigma[2][2];// * 1.5;                        // d/dCzz
}
