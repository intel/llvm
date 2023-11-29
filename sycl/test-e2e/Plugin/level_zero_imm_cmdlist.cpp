// REQUIRES: linux, gpu-intel-pvc, level_zero, level_zero_dev_kit
//
// RUN: %{build} %level_zero_options -o %t.out
// RUN: env UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s
//
// CHECK-NOT: zeCommandListCreate(
// CHECK: zeCommandListCreateImmediate(

// This test checks that immediate commandlists are used and not regular ones on
// PVC Linux.

#include <sycl/sycl.hpp>

int main(int argc, char **argv) {
  sycl::queue Q;
  const unsigned n_chunk = 1000;
  for (int i = 0; i < n_chunk; i++)
    Q.single_task([=]() {});
  Q.wait();
  return 0;
}
