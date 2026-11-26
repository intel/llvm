// REQUIRES: hip
// The amd_arch lit parameter overrides the architectures of the RUN lines.
// UNSUPPORTED: hip-arch-gfx950
// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only \
// RUN:   -fsycl-targets=amd_gpu_gfx90a %s 2>&1 | FileCheck %s
// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only \
// RUN:   -fsycl-targets=amd_gpu_gfx942 %s 2>&1 | FileCheck %s

// gfx950 doubles K for fp16/bf16 (16x16x32, 32x32x16) and int8 (16x16x64,
// 32x32x32). Other AMD targets must reject those fragments: joint_matrix_mad
// has no lowering for them and would otherwise leave the accumulator
// unwritten.

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
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<{{.*}}half, {{.*}}use::a, 16, 32,
  declare_fragment<half, 16, 32, use::a>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<{{.*}}half, {{.*}}use::b, 16, 32,
  declare_fragment<half, 16, 32, use::b>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<{{.*}}bfloat16, {{.*}}use::a, 32, 16,
  declare_fragment<sycl::ext::oneapi::bfloat16, 32, 16, use::a>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<{{.*}}bfloat16, {{.*}}use::b, 32, 16,
  declare_fragment<sycl::ext::oneapi::bfloat16, 32, 16, use::b>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<signed char, {{.*}}use::a, 16, 64,
  declare_fragment<int8_t, 16, 64, use::a>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<signed char, {{.*}}use::b, 64, 16,
  declare_fragment<int8_t, 64, 16, use::b>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<signed char, {{.*}}use::a, 32, 32,
  declare_fragment<int8_t, 32, 32, use::a>(q);
  // CHECK-DAG: error: implicit instantiation of undefined template {{.*}}joint_matrix_hip<signed char, {{.*}}use::b, 32, 32,
  declare_fragment<int8_t, 32, 32, use::b>(q);
  return 0;
}
