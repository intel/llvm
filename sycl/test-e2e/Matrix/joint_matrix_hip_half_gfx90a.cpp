// RUN: %{build} -fsycl -fsycl-targets=amd_gpu_gfx90a -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 %s -o %t.out
// RUN: %{run} %t.out

// REQUIRES: gpu-amd-gfx90a
// REQUIRES: aspect-fp16

#include "joint_matrix_hip_apply.hpp"
#include "joint_matrix_hip_fill.hpp"
#include "joint_matrix_hip_mfma.hpp"

int main() {
  hip_matrix_fill<sycl::half, float, 32, 32, 8, layout::row_major>();
  hip_matrix_fill<sycl::half, float, 16, 16, 16, layout::row_major>();
  hip_matrix_fill<sycl::half, float, 32, 32, 8, layout::col_major>();
  hip_matrix_fill<sycl::half, float, 16, 16, 16, layout::col_major>();

  hip_matrix_fill<sycl::half, float, 32, 32, 8>();
  hip_matrix_fill<sycl::half, float, 16, 16, 16>();

  hip_matrix_apply<sycl::half, float, 32, 32, 8>();
  hip_matrix_apply<sycl::half, float, 16, 16, 16>();
}
