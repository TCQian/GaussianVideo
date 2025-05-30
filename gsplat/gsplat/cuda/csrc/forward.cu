#include "forward.cuh"
#include "helpers.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include "float6.h"
#include <cstdio>

namespace cg = cooperative_groups;

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
__global__ void project_gaussians_forward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* __restrict__ viewmat,
    const float* __restrict__ projmat,
    const float4 intrins,
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,
    float* __restrict__ covs3d,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    float3 p_world = means3d[idx];
    // printf("p_world %d %.2f %.2f %.2f\n", idx, p_world.x, p_world.y,
    // p_world.z);
    float3 p_view;
    if (clip_near_plane(p_world, viewmat, p_view, clip_thresh)) {
        // printf("%d is out of frustum z %.2f, returning\n", idx, p_view.z);
        return;
    }
    // printf("p_view %d %.2f %.2f %.2f\n", idx, p_view.x, p_view.y, p_view.z);

    // compute the projected covariance
    float3 scale = scales[idx];
    float4 quat = quats[idx];
    // printf("%d scale %.2f %.2f %.2f\n", idx, scale.x, scale.y, scale.z);
    // printf("%d quat %.2f %.2f %.2f %.2f\n", idx, quat.w, quat.x, quat.y,
    // quat.z);
    float *cur_cov3d = &(covs3d[6 * idx]);
    scale_rot_to_cov3d(scale, glob_scale, quat, cur_cov3d);

    // project to 2d with ewa approximation
    float fx = intrins.x;
    float fy = intrins.y;
    float cx = intrins.z;
    float cy = intrins.w;
    float tan_fovx = 0.5 * img_size.x / fx;
    float tan_fovy = 0.5 * img_size.y / fy;
    float3 cov2d = project_cov3d_ewa(
        p_world, cur_cov3d, viewmat, fx, fy, tan_fovx, tan_fovy
    );
    // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);

    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok)
        return; // zero determinant
    // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
    conics[idx] = conic;

    // compute the projected mean
    float2 center = project_pix(projmat, p_world, img_size, {cx, cy});
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        // printf("%d point bbox outside of bounds\n", idx);
        return;
    }

    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    xys[idx] = center;
    // printf(
    //     "point %d x %.2f y %.2f z %.2f, radius %d, # tiles %d, tile_min %d
    //     %d, tile_max %d %d\n", idx, center.x, center.y, depths[idx],
    //     radii[idx], tile_area, tile_min.x, tile_min.y, tile_max.x, tile_max.y
    // );
}

// kernel to map each intersection from tile ID and depth to a gaussian
// writes output to isect_ids and gaussian_ids
__global__ void map_gaussian_to_intersects(
    const int num_points,
    const float2* __restrict__ xys,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const int32_t* __restrict__ cum_tiles_hit,
    const dim3 tile_bounds,
    const bool print, // printf or not
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points)
        return;
    if (radii[idx] <= 0)
        return;
    // get the tile bbox for gaussian
    uint2 tile_min, tile_max;
    float2 center = xys[idx];
    get_tile_bbox(center, radii[idx], tile_bounds, tile_min, tile_max);
    
    // print first 3 gaussian's info
    if (print && (idx < 3)) {
        printf("Gaussian %d: center (%f, %f), radius %d, tile_min (%d, %d), tile_max (%d, %d)\n",
               idx, center.x, center.y, radii[idx], tile_min.x, tile_min.y,
               tile_max.x, tile_max.y);
    }
    
    // update the intersection info for all tiles this gaussian hits
    int32_t cur_idx = (idx == 0) ? 0 : cum_tiles_hit[idx - 1];
    // printf("point %d starting at %d\n", idx, cur_idx);
    int64_t depth_id = (int64_t) * (int32_t *)&(depths[idx]);
    for (int i = tile_min.y; i < tile_max.y; ++i) {
        for (int j = tile_min.x; j < tile_max.x; ++j) {
            // isect_id is tile ID and depth as int32
            int64_t tile_id = i * tile_bounds.x + j; // tile within image
            isect_ids[cur_idx] = (tile_id << 32) | depth_id; // tile | depth id
            gaussian_ids[cur_idx] = idx;                     // 3D gaussian id
            ++cur_idx; // handles gaussians that hit more than one tile
        }
    }
    // printf("point %d ending at %d\n", idx, cur_idx);
}

// kernel to map sorted intersection IDs to tile bins
// expect that intersection IDs are sorted by increasing tile ID
// i.e. intersections of a tile are in contiguous chunks
__global__ void get_tile_bin_edges(
    const int num_intersects, const int64_t* __restrict__ isect_ids_sorted, int2* __restrict__ tile_bins
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_intersects)
        return;
    // save the indices where the tile_id changes
    int32_t cur_tile_idx = (int32_t)(isect_ids_sorted[idx] >> 32);
    if (idx == 0 || idx == num_intersects - 1) {
        if (idx == 0)
            tile_bins[cur_tile_idx].x = 0;
        if (idx == num_intersects - 1)
            tile_bins[cur_tile_idx].y = num_intersects;
    }
    if (idx == 0)
        return;
    int32_t prev_tile_idx = (int32_t)(isect_ids_sorted[idx - 1] >> 32);
    if (prev_tile_idx != cur_tile_idx) {
        tile_bins[prev_tile_idx].y = idx;
        tile_bins[cur_tile_idx].x = idx;
        return;
    }
}

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
__global__ void nd_rasterize_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float* __restrict__ out_img,
    const float* __restrict__ background
) {
    // current naive implementation where tile data loading is redundant
    // TODO tile data should be shared between tile threads
    int32_t tile_id = blockIdx.y * tile_bounds.x + blockIdx.x;
    unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    if (i >= img_size.y || j >= img_size.x) {
        return;
    }

    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    float T = 1.f;

    // iterate over all gaussians and apply rendering EWA equation (e.q. 2 from
    // paper)
    int idx;
    for (idx = range.x; idx < range.y; ++idx) {
        const int32_t g = gaussian_ids_sorted[idx];
        const float3 conic = conics[g];
        const float2 center = xys[g];
        const float2 delta = {center.x - px, center.y - py};

        // Mahalanobis distance (here referred to as sigma) measures how many
        // standard deviations away distance delta is. sigma = -0.5(d.T * conic
        // * d)
        const float sigma =
            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
            conic.y * delta.x * delta.y;
        if (sigma < 0.f) {
            continue;
        }
        const float opac = opacities[g];

        const float alpha = min(0.999f, opac * __expf(-sigma));

        // break out conditions
        if (alpha < 1.f / 255.f) {
            continue;
        }
        const float next_T = T * (1.f - alpha);
        if (next_T <= 1e-4f) {
            // we want to render the last gaussian that contributes and note
            // that here idx > range.x so we don't underflow
            idx -= 1;
            break;
        }
        const float vis = alpha * T;
        for (int c = 0; c < channels; ++c) {
            out_img[channels * pix_id + c] += colors[channels * g + c] * vis;
        }
        T = next_T;
    }
    final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
    final_index[pix_id] =
        (idx == range.y)
            ? idx - 1
            : idx; // index of in bin of last gaussian in this pixel
    for (int c = 0; c < channels; ++c) {
        out_img[channels * pix_id + c] += T * background[c];
    }
}



__global__ void rasterize_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    const float3& __restrict__ background
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];

    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            const float alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const float next_T = T * (1.f - alpha);
            if (next_T <= 1e-4f) { // this pixel is done
                // we want to render the last gaussian that contributes and note
                // that here idx > range.x so we don't underflow
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float3 c = colors[g];
            pix_out.x = pix_out.x + c.x * vis;
            pix_out.y = pix_out.y + c.y * vis;
            pix_out.z = pix_out.z + c.z * vis;
            T = next_T;
            cur_idx = batch_start + t;
        }
    }

    if (inside) {
        // add background
        final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
        final_index[pix_id] =
            cur_idx; // index of in bin of last gaussian in this pixel
        float3 final_color;
        final_color.x = pix_out.x + T * background.x;
        final_color.y = pix_out.y + T * background.y;
        final_color.z = pix_out.z + T * background.z;
        out_img[pix_id] = final_color;
    }
}

__global__ void rasterize_video_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const float time,
    const float vis_thresold,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    const float* __restrict__ means_t,
    const float* __restrict__ lambda,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    const float3& __restrict__ background
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float2 time_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];

    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            time_batch[tr] = {lambda[g_id], means_t[g_id]};
            conic_batch[tr] = conics[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float2 time_params = time_batch[t];
            const float3 delta = {xy_opac.x - px, xy_opac.y - py, time - time_params.y};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
        
            const float decay = 0.5 * time_params.x * delta.z * delta.z;
            const float alpha = min(0.999f, opac * __expf(-sigma-decay));
            if (sigma < 0.f || alpha < 1.f / 255.f || decay > vis_thresold) {
                continue;
            }

            const float next_T = T * (1.f - alpha);
            if (next_T <= 1e-4f) { // this pixel is done
                // we want to render the last gaussian that contributes and note
                // that here idx > range.x so we don't underflow
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float3 c = colors[g];
            pix_out.x = pix_out.x + c.x * vis;
            pix_out.y = pix_out.y + c.y * vis;
            pix_out.z = pix_out.z + c.z * vis;
            T = next_T;
            cur_idx = batch_start + t;
        }
    }

    if (inside) {
        // add background
        final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
        final_index[pix_id] =
            cur_idx; // index of in bin of last gaussian in this pixel
        float3 final_color;
        final_color.x = pix_out.x + T * background.x;
        final_color.y = pix_out.y + T * background.y;
        final_color.z = pix_out.z + T * background.z;
        out_img[pix_id] = final_color;
    }
}



// tile_bounds: Dimensions of the tile grid on the 2D image plane.
// img_size: Dimensions of the output image.
// gaussian_ids_sorted: Indices of Gaussians, sorted for efficient processing.
// tile_bins: Specifies the range of Gaussian IDs relevant to each tile.
// xys: 2D positions (centers) of the Gaussians.
// conics: Parameters defining the shape and orientation of each Gaussian.
// colors: RGB color values for each Gaussian.
// opacities: Opacity values for each Gaussian.
// Outputs:
// final_Ts: Stores transmittance for each pixel, though it’s set to T here without updating.
// final_index: Stores the index of the last contributing Gaussian for each pixel.
// out_img: Final rendered image containing each pixel’s RGB color.
// background: Background color applied to transparent areas (though not used here).
__global__ void rasterize_forward_sum(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    const float3& __restrict__ background
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];

    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            const float alpha = min(1.f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            int32_t g = id_batch[t];
            const float vis = alpha;
            const float3 c = colors[g];
            pix_out.x = pix_out.x + c.x * vis;
            pix_out.y = pix_out.y + c.y * vis;
            pix_out.z = pix_out.z + c.z * vis;
            // T = next_T;
            cur_idx = batch_start + t;
        }
        done = true;
    }

    if (inside) {
        // add background
        final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
        final_index[pix_id] =
            cur_idx; // index of in bin of last gaussian in this pixel
        float3 final_color;
        final_color.x = pix_out.x; //+ T * background.x;
        final_color.y = pix_out.y; //+ T * background.y;
        final_color.z = pix_out.z; //+ T * background.z;
        out_img[pix_id] = final_color;
    }
}


__global__ void nd_rasterize_forward_sum(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float* __restrict__ out_img,
    const float* __restrict__ background
) {
    // current naive implementation where tile data loading is redundant
    // TODO tile data should be shared between tile threads
    int32_t tile_id = blockIdx.y * tile_bounds.x + blockIdx.x;
    unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    if (i >= img_size.y || j >= img_size.x) {
        return;
    }

    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    float T = 1.f;

    // iterate over all gaussians and apply rendering EWA equation (e.q. 2 from
    // paper)
    int idx;
    for (idx = range.x; idx < range.y; ++idx) {
        const int32_t g = gaussian_ids_sorted[idx];
        const float3 conic = conics[g];
        const float2 center = xys[g];
        const float2 delta = {center.x - px, center.y - py};

        // Mahalanobis distance (here referred to as sigma) measures how many
        // standard deviations away distance delta is. sigma = -0.5(d.T * conic
        // * d)
        const float sigma =
            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
            conic.y * delta.x * delta.y;
        if (sigma < 0.f) {
            continue;
        }
        const float opac = opacities[g];

        const float alpha = min(1.f, opac * __expf(-sigma));

        // break out conditions
        if (alpha < 1.f / 255.f) {
            continue;
        }
        // const float next_T = T * (1.f - alpha);
        // if (next_T <= 1e-4f) {
        //     // we want to render the last gaussian that contributes and note
        //     // that here idx > range.x so we don't underflow
        //     idx -= 1;
        //     break;
        // }
        const float vis = alpha; //* T;
        for (int c = 0; c < channels; ++c) {
            out_img[channels * pix_id + c] += colors[channels * g + c] * vis;
        }
        //T = next_T;
    }
    final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
    final_index[pix_id] =
        (idx == range.y)
            ? idx - 1
            : idx; // index of in bin of last gaussian in this pixel
    // for (int c = 0; c < channels; ++c) {
    //     out_img[channels * pix_id + c] += T * background[c];
    // }
}


__global__ void rasterize_forward_sum_general(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    const float* __restrict__ betas,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    const float3& __restrict__ background
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];
    __shared__ float beta_batch[BLOCK_SIZE];

    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
            beta_batch[tr] = betas[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float conicPart = conic.x * delta.x * delta.x + conic.z * delta.y * delta.y + 2.f * conic.y * delta.x * delta.y;
            const float beta = beta_batch[t];
            const float sigma = 0.5f * pow(conicPart, beta/2);
            const float opac = xy_opac.z;
            const float alpha = min(1.f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }
            int32_t g = id_batch[t];
            const float vis = alpha;
            const float3 c = colors[g];
            pix_out.x = pix_out.x + c.x * vis;
            pix_out.y = pix_out.y + c.y * vis;
            pix_out.z = pix_out.z + c.z * vis;
            // T = next_T;
            cur_idx = batch_start + t;
        }
        done = true;
    }

    if (inside) {
        // add background
        final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
        final_index[pix_id] =
            cur_idx; // index of in bin of last gaussian in this pixel
        float3 final_color;
        final_color.x = pix_out.x; //+ T * background.x;
        final_color.y = pix_out.y; //+ T * background.y;
        final_color.z = pix_out.z; //+ T * background.z;
        out_img[pix_id] = final_color;
    }
}

// device helper to approximate projected 2d cov from 3d mean and cov
__device__ float3 project_cov3d_ewa(
    const float3& __restrict__ mean3d,
    const float* __restrict__ cov3d,
    const float* __restrict__ viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy
) {
    // clip the
    // we expect row major matrices as input, glm uses column major
    // upper 3x3 submatrix
    glm::mat3 W = glm::mat3(
        viewmat[0],
        viewmat[4],
        viewmat[8],
        viewmat[1],
        viewmat[5],
        viewmat[9],
        viewmat[2],
        viewmat[6],
        viewmat[10]
    );
    glm::vec3 p = glm::vec3(viewmat[3], viewmat[7], viewmat[11]);
    glm::vec3 t = W * glm::vec3(mean3d.x, mean3d.y, mean3d.z) + p;

    // clip so that the covariance
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    t.x = t.z * std::min(lim_x, std::max(-lim_x, t.x / t.z));
    t.y = t.z * std::min(lim_y, std::max(-lim_y, t.y / t.z));

    float rz = 1.f / t.z;
    float rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    glm::mat3 J = glm::mat3(
        fx * rz,
        0.f,
        0.f,
        0.f,
        fy * rz,
        0.f,
        -fx * t.x * rz2,
        -fy * t.y * rz2,
        0.f
    );
    glm::mat3 T = J * W;

    glm::mat3 V = glm::mat3(
        cov3d[0],
        cov3d[1],
        cov3d[2],
        cov3d[1],
        cov3d[3],
        cov3d[4],
        cov3d[2],
        cov3d[4],
        cov3d[5]
    );

    glm::mat3 cov = T * V * glm::transpose(T);

    // add a little blur along axes and save upper triangular elements
    return make_float3(float(cov[0][0]) + 0.3f, float(cov[0][1]), float(cov[1][1]) + 0.3f);
}

// device helper to get 3D covariance from scale and quat parameters
__device__ void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, float *cov3d
) {
    // printf("quat %.2f %.2f %.2f %.2f\n", quat.x, quat.y, quat.z, quat.w);
    glm::mat3 R = quat_to_rotmat(quat);
    // printf("R %.2f %.2f %.2f\n", R[0][0], R[1][1], R[2][2]);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    // printf("S %.2f %.2f %.2f\n", S[0][0], S[1][1], S[2][2]);

    glm::mat3 M = R * S;
    glm::mat3 tmp = M * glm::transpose(M);
    // printf("tmp %.2f %.2f %.2f\n", tmp[0][0], tmp[1][1], tmp[2][2]);

    // save upper right because symmetric
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}

// kernel to map each intersection from tile ID and depth to a gaussian
// writes output to isect_ids and gaussian_ids
__global__ void map_gaussian_to_intersects_video(
    const int num_points, // 9000 (number of gaussians)
    const float3* __restrict__ xys, // (9000, 3) (center of gaussians)
    const float* __restrict__ depths, // useless variable
    const int* __restrict__ radii, // (9000, 1) (radius of gaussians)
    const int32_t* __restrict__ cum_tiles_hit, // (the cumulative tiles hit array)
    const dim3 tile_bounds, // (120, 68, 50)
    const bool print, // printf or not
    // Outputs
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points)
        return;
    if (radii[idx] <= 0)
        return;
    
    // get the tile bbox for gaussian
    uint3 tile_min, tile_max;
    float3 center = xys[idx];
    get_tile_3d_bbox(center, radii[idx], tile_bounds, tile_min, tile_max);

    // print first 3 gaussian's info
    if (print && (idx < 3)) {
        printf("Gaussian %d: center (%f, %f, %f), radius %d, tile_min (%d, %d, %d), tile_max (%d, %d, %d)\n",
               idx, center.x, center.y, center.z, radii[idx], tile_min.x, tile_min.y, tile_min.z,
               tile_max.x, tile_max.y, tile_max.z);
    }
    
   // Update the intersection info for all tiles this Gaussian hits
    int32_t cur_idx = (idx == 0) ? 0 : cum_tiles_hit[idx - 1];
    int64_t depth_id = (int64_t) * (int32_t *)&(depths[idx]);

    // Iterate over the 3D tile bounding box
    for (int k = tile_min.z; k < tile_max.z; ++k) { // Added loop for z dimension
        for (int i = tile_min.y; i < tile_max.y; ++i) {
            for (int j = tile_min.x; j < tile_max.x; ++j) {
                // Calculate the tile ID in 3D
                int64_t tile_id = (k * tile_bounds.y * tile_bounds.x) + (i * tile_bounds.x) + j; // 3D tile index
                isect_ids[cur_idx] = (tile_id << 32) | depth_id; // tile | depth id
                gaussian_ids[cur_idx] = idx;                     // 3D Gaussian id
                ++cur_idx; // Handles Gaussians that hit more than one tile
            }
        }
    }
}

__global__ void rasterize_forward_sum_video(
    const dim3 tile_bounds,              // 3D tile grid dimensions
    const dim3 img_size,                 // 3D image dimensions (width, height, depth)
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,  // Gaussian ranges for each tile. For example [0, 1] in index 0 means the tile at index 0 intersects Gaussians 0 to 1
    const float3* __restrict__ xyzs,     // 3D positions (x, y, z) for each Gaussian
    const float6* __restrict__ conics,   // Shape parameters for each Gaussian
    const float3* __restrict__ colors,   // Colors (R, G, B) for each Gaussian
    const float* __restrict__ opacities, // Opacity for each Gaussian
    float* __restrict__ final_Ts,        // Final transmittance for each voxel
    int* __restrict__ final_index,       // Index of the last Gaussian affecting each voxel
    float3* __restrict__ out_img,        // Output 3D image colors
    const float3& __restrict__ background, // Background color
    const bool print // printf or not
) {
    // printf("Starting rasterize_forward_sum_video\n");
    // Map each thread to a specific voxel within a tile
    auto block = cg::this_thread_block();
    int32_t tile_id = block.group_index().z * tile_bounds.y * tile_bounds.x +
                      block.group_index().y * tile_bounds.x +
                      block.group_index().x;

    unsigned i = block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned k = block.group_index().z * block.group_dim().z + block.thread_index().z;

    float px = (float)j;
    float py = (float)i;
    float pz = (float)k;
    int32_t pix_id = (i * img_size.x + j) + (k * img_size.x * img_size.y);

    // Bounds checking for each voxel in the 3D image
    bool inside = (i < img_size.y && j < img_size.x && k < img_size.z);
    bool done = !inside;

    // Get the Gaussian range for this 3D tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Shared memory for batches
    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float4 xyz_opacity_batch[BLOCK_SIZE];  // Stores x, y, z, and opacity
    __shared__ float6 conic_batch[BLOCK_SIZE];

    // Initialize transmittance and pixel output color
    float T = 1.f;
    int cur_idx = 0;
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};

    // Initialize counter for skipped Gaussians
    int local_total = 0;
    int local_skipped_sigma = 0;
    int local_skipped_alpha = 0;

    // average of delta_x, delta_y, delta_z, and conic parameters
    float avg_delta_x = 0.f;
    float avg_delta_y = 0.f;
    float avg_delta_z = 0.f;
    float avg_conic_x = 0.f;
    float avg_conic_y = 0.f;
    float avg_conic_z = 0.f;
    float avg_conic_w = 0.f;
    float avg_conic_u = 0.f;
    float avg_conic_v = 0.f;

    // Loop through Gaussian batches for this tile
    for (int b = 0; b < num_batches; ++b) {
        // Sync threads before loading the next batch
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {  // Ensure within this tile’s Gaussian range
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float3 xyz = xyzs[g_id];
            const float opac = opacities[g_id];
            xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};  // Store position and opacity
            conic_batch[tr] = conics[g_id];
        }
        
        // Synchronize threads after loading batch
        block.sync();

        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float6 conic = conic_batch[t];
            const float4 xyz_opac = xyz_opacity_batch[t];
            const float opac = xyz_opac.w;
            const float3 delta = {xyz_opac.x - px, xyz_opac.y - py, xyz_opac.z - pz};
            
            // Compute full Mahalanobis distance (times 0.5 factor):
            const float sigma = 0.5f * (
                conic.x * delta.x * delta.x +
                2.f * conic.y * delta.x * delta.y +
                2.f * conic.z * delta.x * delta.z +
                conic.w * delta.y * delta.y +
                2.f * conic.u * delta.y * delta.z +
                conic.v * delta.z * delta.z
            );

            const float alpha = min(1.f, opac * __expf(-sigma));

            // // If voxel (i, j, k) equals (0, 0, 0), print detailed debug info:
            // if (i == 0 && j == 0 && k == 0) {
            //     printf("t=%d: conic=(%.6f, %.6f, %.6f), xyz_opac=(%.6f, %.6f, %.6f, %.6f), opac=%.6f, delta=(%.6f, %.6f, %.6f), sigma=%.6f, alpha=%.6f\n",
            //         t,
            //         conic.x, conic.y, conic.z,
            //         xyz_opac.x, xyz_opac.y, xyz_opac.z, xyz_opac.w,
            //         opac,
            //         delta.x, delta.y, delta.z,
            //         sigma, alpha);
            // }

            local_total++;

            if (sigma < 0.f) {
                local_skipped_sigma++;
                // continue;
            }
            if (alpha < 1.f / 255.f) {
                local_skipped_alpha++;
                // continue;
            }

            if (sigma < 0.f || alpha < 1.f / 255.f) {
                // printf("[DEBUG] Gaussians at i=%u, j=%u, k=%u are skipped due to sigma=%.6f or alpha=%.6f\n", i, j, k, sigma, alpha);
                avg_delta_x += delta.x;
                avg_delta_y += delta.y;
                avg_delta_z += delta.z;
                avg_conic_x += conic.x;
                avg_conic_y += conic.y;
                avg_conic_z += conic.z;
                avg_conic_w += conic.w;
                avg_conic_u += conic.u;
                avg_conic_v += conic.v;
                continue;
            }

            int32_t g = id_batch[t];
            const float vis = alpha;
            const float3 c = colors[g];
            pix_out.x = pix_out.x + c.x * vis;
            pix_out.y = pix_out.y + c.y * vis;
            pix_out.z = pix_out.z + c.z * vis;
            cur_idx = batch_start + t;
        }
        done = true;
    }

    if (inside) {
        // Finalize voxel color with background and store in output
        final_Ts[pix_id] = T; // Remaining transmittance
        final_index[pix_id] = cur_idx; // Index of the last Gaussian affecting this voxel
        float3 final_color;
        final_color.x = pix_out.x;
        final_color.y = pix_out.y;
        final_color.z = pix_out.z;
        out_img[pix_id] = final_color;
        // printf("[DEBUG] Inside voxel (i=%u, j=%u, k=%u), color=(%.2f, %.2f, %.2f)\n",
        //        i, j, k, final_color.x, final_color.y, final_color.z);
    } else {
        if (print && tile_id == 670) {
            int total_skipped = local_skipped_sigma + local_skipped_alpha;
            // print n/total for sigma<0 and m/total for alpha<1/255 and the average deltas and conics
            printf("Tile %d: i=%u, j=%u, k=%u, total=%d, skipped_sigma=%d, skipped_alpha=%d\n, avg_delta=(%.6f, %.6f, %.6f), avg_conic=(%.6f, %.6f, %.6f, %.6f, %.6f, %.6f)\n",
                   tile_id, i, j, k, local_total, local_skipped_sigma, local_skipped_alpha, 
                   avg_delta_x / total_skipped,
                   avg_delta_y / total_skipped,
                   avg_delta_z / total_skipped,
                   avg_conic_x / total_skipped,
                   avg_conic_y / total_skipped,
                   avg_conic_z / total_skipped,
                   avg_conic_w / total_skipped,
                   avg_conic_u / total_skipped,
                   avg_conic_v / total_skipped); 
        }
        // printf("Outside voxel (i=%u, j=%u, k=%u)\n", i, j, k);
    }
}

