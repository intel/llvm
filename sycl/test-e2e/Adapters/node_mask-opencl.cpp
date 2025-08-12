// REQUIRES: aspect-ext_intel_device_info_node_mask
// REQUIRES: gpu, opencl, windows

// RUN: %{build} -o %t.out %opencl_lib
// RUN: %{run} %t.out

// Test that the node mask is read correctly from OpenCL.

#include <iomanip>
#include <iostream>
#include <sstream>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>

int main() {
  sycl::device dev;
  auto nodeMaskSYCL = dev.get_info<sycl::ext::intel::info::device::node_mask>();

  std::cout << "SYCL: " << nodeMaskSYCL << std::endl;

  auto openclDevice = sycl::get_native<sycl::backend::opencl>(dev);

  uint32_t nodeMaskOpencl = 0;

  clGetDeviceInfo(openclDevice, CL_DEVICE_NODE_MASK_KHR, sizeof(uint32_t),
                  &nodeMaskOpencl, nullptr);

  std::cout << "OpenCL  : " << nodeMaskOpencl << std::endl;

  if (nodeMaskSYCL != nodeMaskOpencl) {
    std::cout << "FAILED" << std::endl;
    return -1;
  }

  std::cout << "PASSED" << std::endl;
  return 0;
}
