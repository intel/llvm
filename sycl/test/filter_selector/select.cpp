// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %t1.out

//==------------------- select.cpp - string_selector test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace cl::sycl;
using namespace cl::sycl::ONEAPI;

int main() {
  std::vector<device> CPUs;
  std::vector<device> GPUs;
  std::vector<device> Devs;
  device host;

  CPUs = device::get_devices(info::device_type::cpu);
  GPUs = device::get_devices(info::device_type::gpu);
  Devs = device::get_devices();

  std::cout << "# CPU Devices found: " << CPUs.size() << std::endl;
  std::cout << "# GPU Devices found: " << GPUs.size() << std::endl;
  std::cout << "# Devices found: " << Devs.size() << std::endl;

  bool HasLevelZero = false;
  bool HasOpenCL = false;

  auto Platforms = platform::get_platforms();
  for (auto Platform : Platforms) {
    if (!Platform.is_host()) {
      auto Backend = Platform.get_backend();
      if (Backend == backend::level_zero) {
        HasLevelZero = true;
      } else if (Backend == backend::opencl) {
        HasOpenCL = true;
      }
    }
  }

  if (!CPUs.empty()) {
    std::cout << "Test 'cpu'" << std::endl;
    device d1(filter_selector("cpu"));
    assert(d1.is_cpu());
  }

  if (!GPUs.empty()) {
    std::cout << "Test 'gpu'" << std::endl;
    device d2(filter_selector("gpu"));
    assert(d2.is_gpu());

    std::cout << "Test 'cpu,gpu'" << std::endl;
    device d3(filter_selector("cpu,gpu"));
    assert( (d3.is_gpu() || d3.is_cpu()) );
  }

  if (HasOpenCL) {
    std::cout << "Test 'opencl'" << std::endl;
    device d4(filter_selector("opencl"));
    assert (d4.get_platform().get_backend() == backend::opencl);

    if (!CPUs.empty()) {
      std::cout << "Test 'opencl:cpu'" << std::endl;
      device d5(filter_selector("opencl:cpu"));
      assert(d5.is_cpu() && d5.get_platform().get_backend() == backend::opencl);

      std::cout << "Test 'opencl:cpu:0'" << std::endl;
      device d6(filter_selector("opencl:cpu:0"));
      assert(d6.is_cpu() && d6.get_platform().get_backend() == backend::opencl);
    }

    if (!GPUs.empty()) {
      std::cout << "Test 'opencl:gpu'" << std::endl;
      device d7(filter_selector("opencl:gpu"));
      assert(d7.is_gpu() && d7.get_platform().get_backend() == backend::opencl);
    }
  }

  device d8(filter_selector("0"));

  try {
    // pick something crazy
    device d9(filter_selector("gpu:999"));
    std::cout << "d9 = " << d9.get_info<info::device::name>() << std::endl;
  } catch (runtime_error) {
    std::cout << "Selector failed as expected! OK" << std::endl;
  }
  return 0;
}
