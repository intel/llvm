// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/aspects.hpp>
#include <sycl/detail/core.hpp>

int main() {
  sycl::queue Queue;
  auto Device = Queue.get_device();
  auto backend = Device.get_platform().get_backend();
  bool Expected = (backend == sycl::backend::opencl ||
                   backend == sycl::backend::ext_oneapi_level_zero) &&
                  Device.is_gpu() &&
                  Device.get_info<sycl::info::device::vendor_id>() == 0x8086;

  if (Device.has(sycl::aspect::ext_intel_esimd) != Expected) {
    std::cout << "Unexpected result from device.has(ext_intel_esimd)"
              << std::endl;
    return 1;
  }
  std::cout << "Passed" << std::endl;
  return 0;
}
