// REQUIRES: hip
// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only \
// RUN:   -fsycl-targets=amd_gpu_gfx90a %s 2>&1 | FileCheck %s

// The fp32 (16x16x4, 32x32x2) and packed int8 (16x16x32, 32x32x16)
// joint_matrix shapes are only lowered to MFMA on gfx942 and the pre-release
// MI300 revisions gfx940 and gfx941. gfx90a must reject them at compile time.

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

template <typename T, size_t Rows, size_t Cols, use Use>
void declare_fragment(queue &q) {
  q.submit([&](handler &cgh) {
    cgh.parallel_for(
        nd_range<2>({1, 64}, {1, 64}),
        [=](nd_item<2> it) [[sycl::reqd_sub_group_size(64)]] {
          joint_matrix<sub_group, T, Use, Rows, Cols, layout::row_major> m;
          (void)m;
        });
  });
}

int main() {
  queue q;
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<float, {{.*}}use::a, 16, 4,
  declare_fragment<float, 16, 4, use::a>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<float, {{.*}}use::b, 4, 16,
  declare_fragment<float, 4, 16, use::b>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<float, {{.*}}use::a, 32, 2,
  declare_fragment<float, 32, 2, use::a>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<float, {{.*}}use::b, 2, 32,
  declare_fragment<float, 2, 32, use::b>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<signed char, {{.*}}use::a, 16, 32,
  declare_fragment<int8_t, 16, 32, use::a>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<signed char, {{.*}}use::b, 32, 16,
  declare_fragment<int8_t, 32, 16, use::b>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<signed char, {{.*}}use::a, 32, 16,
  declare_fragment<int8_t, 32, 16, use::a>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<signed char, {{.*}}use::b, 16, 32,
  declare_fragment<int8_t, 16, 32, use::b>(q);
  return 0;
}
