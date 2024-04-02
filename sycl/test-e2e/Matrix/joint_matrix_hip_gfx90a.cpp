// RUN: %{build} -fsycl -fsycl-targets=amd_gpu_gfx90a %s -o %t.out
// RUN: %{run} %t.out

// REQUIRES: gpu-amd-gfx90a

#include "joint_matrix_hip_apply.hpp"
#include "joint_matrix_hip_copy.hpp"
#include "joint_matrix_hip_fill.hpp"
#include "joint_matrix_hip_mfma.hpp"

template <size_t KX> void matrix_mfma() {
  hip_matrix_mfma<int8_t, int32_t, 32, 32, 8, KX, layout::row_major>();
  hip_matrix_mfma<int8_t, int32_t, 16, 16, 16, KX, layout::row_major>();
  hip_matrix_mfma<bfloat16, float, 32, 32, 8, KX, layout::row_major>();
  hip_matrix_mfma<bfloat16, float, 16, 16, 16, KX, layout::row_major>();
  hip_matrix_mfma<double, double, 16, 16, 4, KX, layout::row_major>();
  hip_matrix_mfma<int8_t, int32_t, 32, 32, 8, KX, layout::col_major>();
  hip_matrix_mfma<int8_t, int32_t, 16, 16, 16, KX, layout::col_major>();
  hip_matrix_mfma<bfloat16, float, 32, 32, 8, KX, layout::col_major>();
  hip_matrix_mfma<bfloat16, float, 16, 16, 16, KX, layout::col_major>();
  hip_matrix_mfma<double, double, 16, 16, 4, KX, layout::col_major>();
}

int main() {
  matrix_mfma<1>();
  matrix_mfma<2>();
  matrix_mfma<3>();
  matrix_mfma<4>();

  hip_matrix_copy<int8_t, int32_t, 32, 32, 8, layout::row_major>();
  hip_matrix_copy<int8_t, int32_t, 16, 16, 16, layout::row_major>();
  hip_matrix_copy<bfloat16, float, 32, 32, 8, layout::row_major>();
  hip_matrix_copy<bfloat16, float, 16, 16, 16, layout::row_major>();
  hip_matrix_copy<double, double, 16, 16, 4, layout::row_major>();
  hip_matrix_copy<int8_t, int32_t, 32, 32, 8, layout::col_major>();
  hip_matrix_copy<int8_t, int32_t, 16, 16, 16, layout::col_major>();
  hip_matrix_copy<bfloat16, float, 32, 32, 8, layout::col_major>();
  hip_matrix_copy<bfloat16, float, 16, 16, 16, layout::col_major>();
  hip_matrix_copy<double, double, 16, 16, 4, layout::col_major>();

  hip_matrix_fill<int8_t, int32_t, 32, 32, 8>();
  hip_matrix_fill<int8_t, int32_t, 16, 16, 16>();
  hip_matrix_fill<bfloat16, float, 32, 32, 8>();
  hip_matrix_fill<bfloat16, float, 16, 16, 16>();
  hip_matrix_fill<double, double, 16, 16, 4>();

  hip_matrix_apply<int8_t, int32_t, 32, 32, 8>();
  hip_matrix_apply<int8_t, int32_t, 16, 16, 16>();
  hip_matrix_apply<bfloat16, float, 32, 32, 8>();
  hip_matrix_apply<bfloat16, float, 16, 16, 16>();
  hip_matrix_apply<double, double, 16, 16, 4>();
}
