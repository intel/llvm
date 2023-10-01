// RUN: %{build} -fsycl -fsycl-targets=amd_gpu_gfx90a -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 %s -o %t.out
// RUN: %{run} %t.out

// REQUIRES: gpu-amd-gfx90a

#include "joint_matrix_hip_mfma.hpp"

int main() {
  hip_matrix_mfma<int8_t, int32_t, 32, 32, 8, layout::row_major>();
  hip_matrix_mfma<int8_t, int32_t, 16, 16, 16, layout::row_major>();
  hip_matrix_mfma<bfloat16, float, 32, 32, 8, layout::row_major>();
  hip_matrix_mfma<bfloat16, float, 16, 16, 16, layout::row_major>();
  hip_matrix_mfma<double, double, 16, 16, 4, layout::row_major>();

  hip_matrix_mfma<int8_t, int32_t, 32, 32, 8, layout::col_major>();
  hip_matrix_mfma<int8_t, int32_t, 16, 16, 16, layout::col_major>();
  hip_matrix_mfma<bfloat16, float, 32, 32, 8, layout::col_major>();
  hip_matrix_mfma<bfloat16, float, 16, 16, 16, layout::col_major>();
  hip_matrix_mfma<double, double, 16, 16, 4, layout::col_major>();
}
