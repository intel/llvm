// UNSUPPORTED: accelerator

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  sycl::device dev = q.get_device();

  sycl::ext::oneapi::experimental::architecture arch = dev.get_info<
      sycl::ext::oneapi::experimental::info::device::architecture>();

  assert(dev.ext_oneapi_architecture_is(arch));

  return 0;
}
