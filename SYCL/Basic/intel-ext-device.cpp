// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=level_zero:gpu %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl:gpu %t.out
//
// REQUIRES: gpu
// UNSUPPORTED: cuda
// UNSUPPORTED: hip
// Temporarily disable on L0 due to fails in CI
// UNSUPPORTED: level_zero

//==--------- intel-ext-device.cpp - SYCL device test ------------==//
//
// Returns the low-level device details.  These are Intel-specific extensions
// that are only supported on Level Zero.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s(name, value)
#endif

int main(int argc, char **argv) {
  // Must be enabled at the beginning of the application
  // to obtain the PCI address
  setenv("SYCL_ENABLE_PCI", "1", 0);

  int pltCount = 1;
  for (const auto &plt : platform::get_platforms()) {
    if (!plt.has(aspect::host)) {
      int devCount = 1;
      int totalEUs = 0;
      int numSlices = 0;
      int numSubslices = 0;
      int numEUsPerSubslice = 0;
      int numHWThreadsPerEU = 0;
      for (const auto &dev : plt.get_devices()) {
        std::cout << "Platform #" << pltCount++ << ":" << std::endl;
        if (dev.has(aspect::gpu)) {
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
          } else {
            std::cout << "Unknown" << std::endl;
          }

          // Use Feature Test macro to see if extensions are supported.
          if (SYCL_EXT_INTEL_DEVICE_INFO >= 1) {

            if (dev.has(aspect::ext_intel_pci_address)) {
              std::cout << "PCI address = "
                        << dev.get_info<info::device::ext_intel_pci_address>()
                        << std::endl;
            }
            if (dev.has(aspect::ext_intel_gpu_eu_count)) {
              totalEUs = dev.get_info<info::device::ext_intel_gpu_eu_count>();
              std::cout << "Number of EUs = " << totalEUs << std::endl;
            }
            if (dev.has(aspect::ext_intel_gpu_eu_simd_width)) {
              int w = dev.get_info<info::device::ext_intel_gpu_eu_simd_width>();
              std::cout << "EU SIMD width = " << w << std::endl;
            }
            if (dev.has(aspect::ext_intel_gpu_slices)) {
              numSlices = dev.get_info<info::device::ext_intel_gpu_slices>();
              std::cout << "Number of slices = " << numSlices << std::endl;
            }
            if (dev.has(aspect::ext_intel_gpu_subslices_per_slice)) {
              numSubslices = dev.get_info<
                  info::device::ext_intel_gpu_subslices_per_slice>();
              std::cout << "Number of subslices per slice = " << numSubslices
                        << std::endl;
            }
            if (dev.has(aspect::ext_intel_gpu_eu_count_per_subslice)) {
              numEUsPerSubslice = dev.get_info<
                  info::device::ext_intel_gpu_eu_count_per_subslice>();
              std::cout << "Number of EUs per subslice = " << numEUsPerSubslice
                        << std::endl;
            }
            if (dev.has(aspect::ext_intel_gpu_hw_threads_per_eu)) {
              numHWThreadsPerEU =
                  dev.get_info<info::device::ext_intel_gpu_hw_threads_per_eu>();
              std::cout << "Number of HW threads per EU = " << numHWThreadsPerEU
                        << std::endl;
            }
            if (dev.has(aspect::ext_intel_max_mem_bandwidth)) {
              // not supported yet
              long m =
                  dev.get_info<info::device::ext_intel_max_mem_bandwidth>();
              std::cout << "Maximum memory bandwidth = " << m << std::endl;
            }
            // This is the only data we can verify.
            if (totalEUs != numSlices * numSubslices * numEUsPerSubslice) {
              std::cout << "Error: EU Count is incorrect!" << std::endl;
              std::cout << "Failed!" << std::endl;
              return 1;
            }
          } // SYCL_EXT_INTEL_DEVICE_INFO
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << "Passed!" << std::endl;
  return 0;
}
