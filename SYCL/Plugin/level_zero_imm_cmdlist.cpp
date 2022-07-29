// REQUIRES: level_zero, level_zero_dev_kit
//
// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=level_zero ZE_DEBUG=1 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
//
// CHECK-NOT: zeCommandListCreate(
// CHECK: zeCommandListCreateImmediate(

// The test checks that immediate commandlists are used and not regular ones.

#include <sycl/sycl.hpp>

int main(int argc, char **argv) {
  sycl::queue Q;
  const unsigned n_chunk = 1000;
  for (int i = 0; i < n_chunk; i++)
    Q.single_task([=]() {});
  Q.wait();
  return 0;
}
