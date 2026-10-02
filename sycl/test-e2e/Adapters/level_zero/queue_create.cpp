// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %{build} %level_zero_options -o %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=%if level_zero_v2_adapter %{CHECK-V2%} %else %{CHECK%}
//
// The test checks that the Level Zero execution resources associated with a
// SYCL queue are created once and reused across all 1000 submissions,
// rather than being recreated per-submission, using the embedded
// UR_L0_LEAKS_DEBUG=1 testing capability.
//
// v1 creates a single regular command queue for the whole run:
// CHECK:  zeCommandQueueCreate = 1     \--->         zeCommandQueueDestroy = 1
//
// L0v2 reuses a fixed pool of 4 immediate command lists.
// CHECK-V2: zeCommandListCreateImmediate = 4
// CHECK-V2: zeCommandListCreate = 0     \---> zeCommandListDestroy = 4
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
