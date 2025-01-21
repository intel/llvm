// REQUIRES: gpu
// REQUIRES: level_zero || opencl
// RUN: %{build} -o %t.out
// ZES_SYSMAN_ENABLE should be set to query free_memory.
// RUN: env ZES_SYSMAN_ENABLE=1 %{run} %t.out
//
//==--------- intel-ext-device.cpp - SYCL device test ------------==//
//
// Returns the low-level device details.  These are Intel-specific extensions
// that are only supported on Level Zero or OpenCL.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>

#include <cassert>
#include <iostream>

using namespace sycl;

int main(int argc, char **argv) {

  int pltCount = 1;
  for (const auto &plt : platform::get_platforms()) {
    int devCount = 1;
    int totalEUs = 0;
    int numSlices = 0;
    int numSubslices = 0;
    int numEUsPerSubslice = 0;
    int numHWThreadsPerEU = 0;
    for (const auto &dev : plt.get_devices()) {
      std::cout << "Platform #" << pltCount++ << ":" << std::endl;
      auto name = dev.get_info<info::device::name>();
      std::cout << "Device #" << devCount++ << ": "
                << dev.get_info<info::device::name>() << ":" << std::endl;

      std::cout << "Backend: ";
      if (plt.get_backend() == backend::ext_oneapi_level_zero) {
        std::cout << "Level Zero" << std::endl;
      } else if (plt.get_backend() == backend::opencl) {
        std::cout << "OpenCL" << std::endl;
      } else if (plt.get_backend() == backend::ext_oneapi_cuda) {
        std::cout << "CUDA" << std::endl;
      } else if (plt.get_backend() == backend::ext_oneapi_hip) {
        std::cout << "HIP" << std::endl;
      } else {
        std::cout << "Unknown" << std::endl;
      }

      // Use Feature Test macro to see if extensions are supported.
      if (SYCL_EXT_INTEL_DEVICE_INFO >= 1) {

        if (dev.has(aspect::ext_intel_pci_address)) {
          std::cout << "PCI address = "
                    << dev.get_info<ext::intel::info::device::pci_address>()
                    << std::endl;
        }
        if (dev.has(aspect::ext_intel_gpu_eu_count)) {
          totalEUs = dev.get_info<ext::intel::info::device::gpu_eu_count>();
          std::cout << "Number of EUs = " << totalEUs << std::endl;
        }
        if (dev.has(aspect::ext_intel_gpu_eu_simd_width)) {
          int w = dev.get_info<ext::intel::info::device::gpu_eu_simd_width>();
          std::cout << "EU SIMD width = " << w << std::endl;
        }
        if (dev.has(aspect::ext_intel_gpu_slices)) {
          numSlices = dev.get_info<ext::intel::info::device::gpu_slices>();
          std::cout << "Number of slices = " << numSlices << std::endl;
        }
        if (dev.has(aspect::ext_intel_gpu_subslices_per_slice)) {
          numSubslices =
              dev.get_info<ext::intel::info::device::gpu_subslices_per_slice>();
          std::cout << "Number of subslices per slice = " << numSubslices
                    << std::endl;
        }
        if (dev.has(aspect::ext_intel_gpu_eu_count_per_subslice)) {
          numEUsPerSubslice = dev.get_info<
              ext::intel::info::device::gpu_eu_count_per_subslice>();
          std::cout << "Number of EUs per subslice = " << numEUsPerSubslice
                    << std::endl;
        }
        if (SYCL_EXT_INTEL_DEVICE_INFO >= 3 &&
            dev.has(aspect::ext_intel_gpu_hw_threads_per_eu)) {
          numHWThreadsPerEU =
              dev.get_info<ext::intel::info::device::gpu_hw_threads_per_eu>();
          std::cout << "Number of HW threads per EU = " << numHWThreadsPerEU
                    << std::endl;
        }
        if (dev.has(aspect::ext_intel_max_mem_bandwidth)) {
          // not supported yet
          long m = dev.get_info<ext::intel::info::device::max_mem_bandwidth>();
          std::cout << "Maximum memory bandwidth = " << m << std::endl;
        }
        // This is the only data we can verify.
        if (totalEUs != numSlices * numSubslices * numEUsPerSubslice) {
          std::cout << "Error: EU Count is incorrect!" << std::endl;
          std::cout << "Failed!" << std::endl;
          return 1;
        }
        if (dev.has(aspect::ext_intel_free_memory)) {
          auto TotalMemory = dev.get_info<info::device::global_mem_size>();
          auto FreeMemory =
              dev.get_info<ext::intel::info::device::free_memory>();
          std::cout << "FreeMemory = " << FreeMemory << std::endl;
          assert((TotalMemory >= FreeMemory) &&
                 "Expect total_memory >= free_memory");
        }
        if (SYCL_EXT_INTEL_DEVICE_INFO >= 2 &&
            dev.has(aspect::ext_intel_device_info_uuid)) {
          auto UUID = dev.get_info<ext::intel::info::device::uuid>();
          std::cout << "Device UUID = ";
          for (int i = 0; i < 16; i++) {
            std::cout << std::to_string(UUID[i]);
          }
          std::cout << "\n";
        }
        if (SYCL_EXT_INTEL_DEVICE_INFO >= 5 &&
            dev.has(aspect::ext_intel_device_id)) {
          int deviceID = dev.get_info<ext::intel::info::device::device_id>();
          std::cout << "Device ID = " << deviceID << std::endl;
        }
      } // SYCL_EXT_INTEL_DEVICE_INFO
      std::cout << std::endl;
    }
  }
  std::cout << "Passed!" << std::endl;
  return 0;
}
