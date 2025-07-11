// REQUIRES: aspect-ext_intel_device_info_luid
// REQUIRES: gpu, cuda, cuda_dev_kit, windows

// RUN: %{build} %cuda_options -o %t.out
// RUN: %{run} %t.out

// Test that the LUID is read correctly from Level Zero.

#include <iomanip>
#include <iostream>
#include <sstream>
#define SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL 1
#include <cuda.h>
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

  CUdevice cudaDevice = sycl::get_native<sycl::backend::ext_oneapi_cuda>(dev);

  char *luidCuda = nullptr;

  cuDeviceGetLuid(luidCuda, nullptr, cudaDevice);

  std::stringstream luidCudaHex;
  for (int i = 0; i < 8; ++i)
    luidCudaHex << std::hex << std::setw(2) << std::setfill('0')
                << int(luidCuda[i]);
  std::cout << "CUDA  : " << luidCudaHex.str() << std::endl;

  if (luidSYCLHex.str() != luidCudaHex.str()) {
    std::cout << "FAILED" << std::endl;
    return -1;
  }

  std::cout << "PASSED" << std::endl;
  return 0;
}
