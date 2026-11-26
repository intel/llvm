// REQUIRES: hip
// The amd_arch lit parameter overrides the architectures of the RUN lines.
// UNSUPPORTED: hip-arch-gfx90a
// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only \
// RUN:   -fsycl-targets=amd_gpu_gfx942 %s 2>&1 | FileCheck %s
// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only \
// RUN:   -fsycl-targets=amd_gpu_gfx950 %s 2>&1 | FileCheck %s

// The int8 16x16x16 and 32x32x8 MFMAs only exist on gfx90a: CDNA3 and CDNA4
// replace them with larger K shapes. Those targets must reject the fragments,
// as joint_matrix_mad has no lowering for them there and would otherwise leave
// the accumulator unwritten.

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
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<signed char, {{.*}}use::a, 16, 16,
  declare_fragment<int8_t, 16, 16, use::a>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<signed char, {{.*}}use::b, 16, 16,
  declare_fragment<int8_t, 16, 16, use::b>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<signed char, {{.*}}use::a, 32, 8,
  declare_fragment<int8_t, 32, 8, use::a>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<signed char, {{.*}}use::b, 8, 32,
  declare_fragment<int8_t, 8, 32, use::b>(q);
  return 0;
}
