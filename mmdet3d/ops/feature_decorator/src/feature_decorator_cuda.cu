#include <stdio.h>
#include <stdlib.h>

// Each thread handles one voxel: copies features and appends decoration dimensions.
__global__ void feature_decorator_kernel(
    int n, int max_pts, int feat_dim, int out_dim,
    const float* __restrict__ features,
    const int* __restrict__ coors,
    const int* __restrict__ num_pts,
    float vx, float vy, float x_offset, float y_offset,
    int use_cluster, int use_center,
    float* __restrict__ out) {
  int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (voxel_idx >= n) return;

  int npts = num_pts[voxel_idx];

  // Compute cluster mean (mean of x, y, z over valid points) when requested.
  float mean_x = 0.0f, mean_y = 0.0f, mean_z = 0.0f;
  if (use_cluster > 0 && npts > 0) {
    for (int i = 0; i < npts; i++) {
      int src = (voxel_idx * max_pts + i) * feat_dim;
      mean_x += features[src + 0];
      mean_y += features[src + 1];
      mean_z += features[src + 2];
    }
    mean_x /= npts;
    mean_y /= npts;
    mean_z /= npts;
  }

  // Voxel center in metric coordinates.
  float voxel_center_x = coors[voxel_idx * 4 + 1] * vx + x_offset;
  float voxel_center_y = coors[voxel_idx * 4 + 2] * vy + y_offset;

  for (int i = 0; i < max_pts; i++) {
    int src = (voxel_idx * max_pts + i) * feat_dim;
    int dst = (voxel_idx * max_pts + i) * out_dim;

    // Copy original features; zero out padded points.
    for (int j = 0; j < feat_dim; j++) {
      out[dst + j] = (i < npts) ? features[src + j] : 0.0f;
    }

    int decoration_offset = feat_dim;

    // Cluster decoration: offset of each point from the cluster mean.
    if (use_cluster > 0) {
      out[dst + decoration_offset + 0] = (i < npts) ? (features[src + 0] - mean_x) : 0.0f;
      out[dst + decoration_offset + 1] = (i < npts) ? (features[src + 1] - mean_y) : 0.0f;
      out[dst + decoration_offset + 2] = (i < npts) ? (features[src + 2] - mean_z) : 0.0f;
      decoration_offset += 3;
    }

    // Center decoration: offset of each point from the voxel pillar center.
    if (use_center > 0) {
      out[dst + decoration_offset + 0] = (i < npts) ? (features[src + 0] - voxel_center_x) : 0.0f;
      out[dst + decoration_offset + 1] = (i < npts) ? (features[src + 1] - voxel_center_y) : 0.0f;
    }
  }
}

void feature_decorator(
    int n, int max_pts, int feat_dim, int out_dim,
    const float* x, const int* y, const int* z,
    float vx, float vy, float x_offset, float y_offset,
    int use_cluster, int use_center, float* out) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  feature_decorator_kernel<<<blocks, threads>>>(
      n, max_pts, feat_dim, out_dim, x, y, z,
      vx, vy, x_offset, y_offset,
      use_cluster, use_center, out);
}

