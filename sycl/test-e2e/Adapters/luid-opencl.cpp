// REQUIRES: aspect-ext_intel_device_info_luid
// REQUIRES: gpu, opencl, windows

// RUN: %{build} -o %t.out %opencl_lib
// RUN: %{run} %t.out

// Test that the LUID is read correctly from OpenCL.

#include <iomanip>
#include <iostream>
#include <sstream>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>

int main() {
  sycl::device dev;
  auto luid = dev.get_info<sycl::ext::intel::info::device::luid>();

  std::stringstream luidSYCLHex;
  for (int i = 0; i < luid.size(); ++i) {
    luidSYCLHex << std::hex << std::setw(2) << std::setfill('0')
                << int(luid[i]);
  }
  std::cout << "SYCL: " << luidSYCLHex.str() << std::endl;

  auto openclDevice = sycl::get_native<sycl::backend::opencl>(dev);

  std::array<unsigned char, CL_LUID_SIZE_KHR> luidOpencl{};

  clGetDeviceInfo(openclDevice, CL_DEVICE_LUID_KHR,
                  sizeof(char) * CL_LUID_SIZE_KHR, luidOpencl.data(), nullptr);

  std::stringstream luidOpenclHex;
  for (int i = 0; i < 8; ++i)
    luidOpenclHex << std::hex << std::setw(2) << std::setfill('0')
                  << int(luidOpencl[i]);
  std::cout << "OpenCL  : " << luidOpenclHex.str() << std::endl;

  if (luidSYCLHex.str() != luidOpenclHex.str()) {
    std::cout << "FAILED" << std::endl;
    return -1;
  }

  std::cout << "PASSED" << std::endl;
  return 0;
}
