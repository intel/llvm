// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

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
