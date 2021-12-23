// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// XFAIL: hip_nvidia
// Issue #106: The test failing sporadically on OpenCL platform due to
// processing OCL_ICD_FILENAMES debug environment variable which causes
// extra memory allocation on device creation.
// Issue #661: The test is failing sporadically on HIP AMD.
// UNSUPPORTED: windows, opencl, hip_amd
//
//==-----memory-consumption.cpp - SYCL memory consumption basic test ------==//
//
// This test specifically tests that memory consumption does not change
// when the get_devices() is called repeatedly.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/sycl.hpp"
#include <iostream>
#include <thread>

#ifdef __linux__
#include <strings.h>
#include <sys/resource.h>
#include <sys/time.h>

long get_cpu_mem() {
  struct rusage usage;
  memset(&usage, 0, sizeof(rusage));
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss;
}
#endif

using namespace cl::sycl;

int main() {
  constexpr auto dev_type = info::device_type::gpu;
  auto devices = device::get_devices(dev_type);

  int startSize = get_cpu_mem();
  std::cout << startSize << " kb" << std::endl;

  for (int i = 0; i < 1000; i++) {
    devices = cl::sycl::device::get_devices(dev_type);
  }
  int endSize = get_cpu_mem();
  std::cout << endSize << " kb" << std::endl;

  auto plat = devices[0].get_platform();
  std::string plat_name = plat.get_info<info::platform::name>();
  std::cout << "Platform: " << plat_name << std::endl;

  devices.erase(devices.begin(), devices.end());

  if (startSize == endSize) {
    std::cout << "Passed" << std::endl;
    return 0;
  } else {
    std::cout << "Failed" << std::endl;
    return 1;
  }
}
