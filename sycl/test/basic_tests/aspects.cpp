// RUN: %clangxx %s -o %t.out -I %sycl_include -lsycl
// RUN: %t.out

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

int main() {
  bool failed = false;
  int pltIdx = 0;
  for (const auto &plt : platform::get_platforms()) {
    pltIdx++;
    if (plt.has(aspect::host)) {
      std::cout << "Platform #" << pltIdx
                << " type: host supports:" << std::endl;
    } else if (plt.has(aspect::cpu)) {
      std::cout << "Platform #" << pltIdx
                << " type: cpu supports:" << std::endl;
    } else if (plt.has(aspect::gpu)) {
      std::cout << "Platform #" << pltIdx
                << " type: gpu supports:" << std::endl;
    } else if (plt.has(aspect::accelerator)) {
      std::cout << "Platform #" << pltIdx
                << " type: accelerator supports:" << std::endl;
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
    if (plt.has(aspect::int64_base_atomics)) {
      std::cout << "  base atomic operations" << std::endl;
    }
    if (plt.has(aspect::int64_extended_atomics)) {
      std::cout << "  extended atomic operations" << std::endl;
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

    int devIdx = 0;
    for (const auto &dev : plt.get_devices()) {
      devIdx++;
      if (dev.has(aspect::host)) {
        if ((!dev.has(aspect::image)) || (!dev.has(aspect::online_compiler)) ||
            (!dev.has(aspect::online_linker)) ||
            (!dev.has(aspect::queue_profiling))) {
          failed = true;
          std::cout << "Failed: device #" << devIdx << " type: host"
                    << std::endl;
        }
      } else if (dev.has(aspect::cpu)) {
        if ((!dev.has(aspect::fp64)) ||
            (!dev.has(aspect::int64_base_atomics)) ||
            (!dev.has(aspect::int64_extended_atomics)) ||
            (!dev.has(aspect::image)) || (!dev.has(aspect::online_compiler)) ||
            (!dev.has(aspect::online_linker)) ||
            (!dev.has(aspect::queue_profiling))) {
          failed = true;
          std::cout << "Failed: device #" << devIdx << " type: cpu"
                    << std::endl;
        }
      } else if (dev.has(aspect::gpu)) {
        if ((!dev.has(aspect::fp16)) || (!dev.has(aspect::fp64)) ||
            (!dev.has(aspect::int64_base_atomics)) ||
            (!dev.has(aspect::int64_extended_atomics)) ||
            (!dev.has(aspect::image)) || (!dev.has(aspect::online_compiler)) ||
            (!dev.has(aspect::online_linker)) ||
            (!dev.has(aspect::queue_profiling))) {
          failed = true;
          std::cout << "Failed: device #" << devIdx << " type: gpu"
                    << std::endl;
        }
      } else if (dev.has(aspect::accelerator)) {
        if ((!dev.has(aspect::online_compiler)) ||
            (!dev.has(aspect::online_linker)) ||
            (!dev.has(aspect::queue_profiling))) {
          failed = true;
          std::cout << "Failed: device #" << devIdx << " type: gpu"
                    << std::endl;
        }
      } else {
        failed = true;
        std::cout << "Failed: device #" << devIdx << " type: unknown"
                  << std::endl;
      }
    }
  }
  if (failed) {
    return 1;
  }
  std::cout << "Passed." << std::endl;
  return 0;
}
