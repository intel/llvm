// REQUIRES: aspect-ext_intel_device_info_node_mask
// REQUIRES: gpu, target-nvidia, cuda_dev_kit, windows

// RUN: %{build} %cuda_options -o %t.out
// RUN: %{run} %t.out

// Test that the node mask is read correctly from CUDA.

#include <iomanip>
#include <iostream>
#include <sstream>
#define SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL 1
#include <cuda.h>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>

int main() {
  sycl::device dev;
  uint32_t nodeMaskSYCL =
      dev.get_info<sycl::ext::intel::info::device::node_mask>();

  std::cout << "SYCL: " << nodeMaskSYCL << std::endl;

  CUdevice cudaDevice = sycl::get_native<sycl::backend::ext_oneapi_cuda>(dev);

  uint32_t nodeMaskCuda = 0;

  cuDeviceGetLuid(nullptr, &nodeMaskCuda, cudaDevice);

  std::cout << "CUDA  : " << nodeMaskCuda << std::endl;

  if (nodeMaskSYCL != nodeMaskCuda) {
    std::cout << "FAILED" << std::endl;
    return -1;
  }

  std::cout << "PASSED" << std::endl;
  return 0;
}
