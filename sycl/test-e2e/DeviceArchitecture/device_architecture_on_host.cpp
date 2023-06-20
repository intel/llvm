// UNSUPPORTED: cuda, hip, esimd_emulator

// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  sycl::device dev = q.get_device();

  sycl::ext::oneapi::experimental::architecture arch = dev.get_info<
      sycl::ext::oneapi::experimental::info::device::architecture>();
  // CHECK: piDeviceGetInfo

  bool is_arch_bdw = dev.ext_oneapi_architecture_is(
      sycl::ext::oneapi::experimental::architecture::intel_gpu_bdw);
  // CHECK: piDeviceGetInfo
  return 0;
}
