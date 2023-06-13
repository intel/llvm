// UNSUPPORTED: cpu, cuda, hip, esimd_emulator

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

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
