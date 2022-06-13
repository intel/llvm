// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=%sycl_be %t.out
//
// Hip is missing some of the parameters tested here so it fails with NVIDIA
// XFAIL: hip_nvidia

// XFAIL: hip
// Expected failure because hip does not have atomic64 check implementation

//==--------------- aspects.cpp - SYCL device test ------------------------==//
//
// Returns the various aspects of a device  and platform.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

// platform::has() calls device::has() for each device on the platform.

int main() {
  bool failed = false;
  int pltIdx = 0;
  for (const auto &plt : platform::get_platforms()) {
    pltIdx++;
    if (plt.has(aspect::host)) {
      std::cout << "Platform #" << pltIdx
                << " type: Host supports:" << std::endl;
    } else if (plt.has(aspect::cpu)) {
      std::cout << "Platform #" << pltIdx
                << " type: CPU supports:" << std::endl;
    } else if (plt.has(aspect::gpu)) {
      std::cout << "Platform #" << pltIdx
                << " type: GPU supports:" << std::endl;
    } else if (plt.has(aspect::accelerator)) {
      std::cout << "Platform #" << pltIdx
                << " type: Accelerator supports:" << std::endl;
    } else if (plt.has(aspect::custom)) {
      std::cout << "Platform #" << pltIdx
                << " type: Custom supports:" << std::endl;
    } else {
      failed = true;
      std::cout << "Failed: platform #" << pltIdx << " type: unknown"
                << std::endl;
      return 1;
    }

    if (plt.has(aspect::fp16)) {
      std::cout << "  fp16" << std::endl;
    }
    if (plt.has(aspect::fp64)) {
      std::cout << "  fp64" << std::endl;
    }
    if (plt.has(aspect::ext_oneapi_bfloat16)) {
      std::cout << " ext_oneapi_bfloat16" << std::endl;
    }
    if (plt.has(aspect::int64_base_atomics)) {
      std::cout << "  base atomic operations" << std::endl;
    }
    if (plt.has(aspect::int64_extended_atomics)) {
      std::cout << "  extended atomic operations" << std::endl;
    }
    if (plt.has(aspect::atomic64)) {
      std::cout << "  atomic64" << std::endl;
    }
    if (plt.has(aspect::image)) {
      std::cout << "  images" << std::endl;
    }
    if (plt.has(aspect::online_compiler)) {
      std::cout << "  online compiler" << std::endl;
    }
    if (plt.has(aspect::online_linker)) {
      std::cout << "  online linker" << std::endl;
    }
    if (plt.has(aspect::queue_profiling)) {
      std::cout << "  queue profiling" << std::endl;
    }
    if (plt.has(aspect::usm_device_allocations)) {
      std::cout << "  USM allocations" << std::endl;
    }
    if (plt.has(aspect::usm_host_allocations)) {
      std::cout << "  USM host allocations" << std::endl;
    }
    if (plt.has(aspect::usm_atomic_host_allocations)) {
      std::cout << "  USM atomic host allocations" << std::endl;
    }
    if (plt.has(aspect::usm_shared_allocations)) {
      std::cout << "  USM shared allocations" << std::endl;
    }
    if (plt.has(aspect::usm_atomic_shared_allocations)) {
      std::cout << "  USM atomic shared allocations" << std::endl;
    }
    if (plt.has(aspect::usm_restricted_shared_allocations)) {
      std::cout << "  USM restricted shared allocations" << std::endl;
    }
    if (plt.has(aspect::usm_system_allocator)) {
      std::cout << "  USM system allocator" << std::endl;
    }
    if (plt.has(aspect::usm_system_allocations)) {
      std::cout << "  USM system allocations" << std::endl;
    }
  }
  std::cout << "Passed." << std::endl;
  return 0;
}
