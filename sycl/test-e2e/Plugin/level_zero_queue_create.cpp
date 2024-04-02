// REQUIRES: level_zero, level_zero_dev_kit

// RUN: %{build} %level_zero_options -o %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s
//
// CHECK:  zeCommandQueueCreate = 1     \--->         zeCommandQueueDestroy = 1
// The test is to check that there is only a single level zero queue created
// with the embedded UR_L0_LEAKS_DEBUG=1 testing capability.
//

#include <sycl/detail/core.hpp>

int main(int argc, char **argv) {
  sycl::queue Q;
  const unsigned n_chunk = 1000;
  for (int i = 0; i < n_chunk; i++)
    Q.single_task([=]() {});
  Q.wait();
  return 0;
}
