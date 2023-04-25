// REQUIRES: level_zero, level_zero_dev_kit

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 ZE_DEBUG=4 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
//
// CHECK:  zeCommandQueueCreate = 1     \--->         zeCommandQueueDestroy = 1
// The test is to check that there is only a single level zero queue created
// with the embedded ZE_DEBUG=4 testing capability.
//

#include <sycl/sycl.hpp>

int main(int argc, char **argv) {
  sycl::queue Q;
  const unsigned n_chunk = 1000;
  for (int i = 0; i < n_chunk; i++)
    Q.single_task([=]() {});
  Q.wait();
  return 0;
}
