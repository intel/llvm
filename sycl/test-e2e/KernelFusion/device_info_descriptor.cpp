// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// XFAIL: cuda || hip
// REQUIRES: fusion

// Test correct return from device information descriptor.

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {

  queue q;

  assert(q.get_device()
             .get_info<sycl::ext::codeplay::experimental::info::device::
                           supports_fusion>() &&
         "Device should support fusion");

  return 0;
}
