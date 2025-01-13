// REQUIRES: accelerator

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test checks that device_architecture extension implementation correctly
// handles unsupported HW. The unsupported HW in this test is any FPGA device,
// as FPGA currently is not supported at all by the device_architecture
// extension.

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;
  sycl::device dev = q.get_device();

  sycl::ext::oneapi::experimental::architecture arch = dev.get_info<
      sycl::ext::oneapi::experimental::info::device::architecture>();

  assert(arch == sycl::ext::oneapi::experimental::architecture::unknown);
  // device::ext_oneapi_architecture_is(syclex::architecture::unknown) should
  // return true if the device does not have a known architecture.
  assert(dev.ext_oneapi_architecture_is(arch));

  // No exceptions are expected in this test.

  return 0;
}
