// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

//==------------------- select.cpp - filter_selector test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;

int main() {
  std::vector<device> CPUs;
  std::vector<device> GPUs;
  std::vector<device> Accels;
  std::vector<device> Devs;
  device host;

  CPUs = device::get_devices(info::device_type::cpu);
  GPUs = device::get_devices(info::device_type::gpu);
  Accels = device::get_devices(info::device_type::accelerator);
  Devs = device::get_devices();

  std::cout << "# CPU Devices found: " << CPUs.size() << std::endl;
  std::cout << "# GPU Devices found: " << GPUs.size() << std::endl;
  std::cout << "# Accelerators found: " << Accels.size() << std::endl;
  std::cout << "# Devices found: " << Devs.size() << std::endl;

  bool HasLevelZeroDevices = false;
  bool HasOpenCLDevices = false;
  bool HasCUDADevices = false;
  bool HasHIPDevices = false;
  bool HasOpenCLGPU = false;
  bool HasLevelZeroGPU = false;

  for (auto &Dev : Devs) {
    auto Backend = Dev.get_platform().get_backend();
    if (Backend == backend::ext_oneapi_level_zero) {
      HasLevelZeroDevices = true;
    } else if (Backend == backend::opencl) {
      HasOpenCLDevices = true;
    } else if (Backend == backend::ext_oneapi_cuda) {
      HasCUDADevices = true;
    } else if (Backend == backend::ext_oneapi_hip) {
      HasHIPDevices = true;
    }
  }

  for (const auto &GPU : GPUs) {
    if (GPU.get_platform().get_backend() == backend::opencl) {
      HasOpenCLGPU = true;
    } else if (GPU.get_platform().get_backend() ==
               backend::ext_oneapi_level_zero) {
      HasLevelZeroGPU = true;
    }
  }

  std::cout << "HasLevelZeroDevices = " << HasLevelZeroDevices << std::endl;
  std::cout << "HasOpenCLDevices    = " << HasOpenCLDevices << std::endl;
  std::cout << "HasCUDADevices      = " << HasCUDADevices << std::endl;
  std::cout << "HasHIPDevices       = " << HasHIPDevices << std::endl;
  std::cout << "HasOpenCLGPU        = " << HasOpenCLGPU << std::endl;
  std::cout << "HasLevelZeroGPU     = " << HasLevelZeroGPU << std::endl;

  if (!CPUs.empty()) {
    std::cout << "Test 'cpu'";
    device d1(filter_selector("cpu"));
    assert(d1.is_cpu());
    std::cout << "...PASS" << std::endl;
  }

  if (!GPUs.empty()) {
    std::cout << "Test 'gpu'";
    device d2(filter_selector("gpu"));
    assert(d2.is_gpu());
    std::cout << "...PASS" << std::endl;
  }

  if (!CPUs.empty() || !GPUs.empty()) {
    std::cout << "Test 'cpu,gpu'";
    device d3(filter_selector("cpu,gpu"));
    assert((d3.is_gpu() || d3.is_cpu()));
    std::cout << "...PASS" << std::endl;
  }

  if (HasOpenCLDevices) {
    std::cout << "Test 'opencl'";
    device d4(filter_selector("opencl"));
    assert(d4.get_platform().get_backend() == backend::opencl);
    std::cout << "...PASS" << std::endl;

    if (!CPUs.empty()) {
      std::cout << "Test 'opencl:cpu'";
      device d5(filter_selector("opencl:cpu"));
      assert(d5.is_cpu() && d5.get_platform().get_backend() == backend::opencl);
      std::cout << "...PASS" << std::endl;

      std::cout << "Test 'opencl:cpu:0'";
      device d6(filter_selector("opencl:cpu:0"));
      assert(d6.is_cpu() && d6.get_platform().get_backend() == backend::opencl);
      std::cout << "...PASS" << std::endl;
    }

    if (HasOpenCLGPU) {
      std::cout << "Test 'opencl:gpu'" << std::endl;
      device d7(filter_selector("opencl:gpu"));
      assert(d7.is_gpu() && d7.get_platform().get_backend() == backend::opencl);
    }
  }

  std::cout << "Test '0'";
  device d8(filter_selector("0"));
  std::cout << "...PASS" << std::endl;

  try {
    // pick something crazy
    device d9(filter_selector("gpu:999"));
    std::cout << "d9 = " << d9.get_info<info::device::name>() << std::endl;
  } catch (const sycl::runtime_error &e) {
    const char *ErrorMesg =
        "Could not find a device that matches the specified filter(s)!";
    assert(std::string{e.what()}.find(ErrorMesg) == 0);
    std::cout << "Selector failed as expected! OK" << std::endl;
  }

  try {
    // pick something crazy
    device d10(filter_selector("bob:gpu"));
    std::cout << "d10 = " << d10.get_info<info::device::name>() << std::endl;
  } catch (const sycl::runtime_error &e) {
    const char *ErrorMesg = "Invalid filter string!";
    assert(std::string{e.what()}.find(ErrorMesg) == 0);
    std::cout << "Selector failed as expected! OK" << std::endl;
  }

  try {
    // pick something crazy
    device d11(filter_selector("opencl:bob"));
    std::cout << "d11 = " << d11.get_info<info::device::name>() << std::endl;
  } catch (const sycl::runtime_error &e) {
    const char *ErrorMesg = "Invalid filter string!";
    assert(std::string{e.what()}.find(ErrorMesg) == 0);
    std::cout << "Selector failed as expected! OK" << std::endl;
  }

  if (HasLevelZeroDevices && HasLevelZeroGPU) {
    std::cout << "Test 'level_zero'";
    device d12(filter_selector("level_zero"));
    assert(d12.get_platform().get_backend() == backend::ext_oneapi_level_zero);
    std::cout << "...PASS" << std::endl;

    std::cout << "Test 'level_zero:gpu'";
    device d13(filter_selector("level_zero:gpu"));
    assert(d13.is_gpu() &&
           d13.get_platform().get_backend() == backend::ext_oneapi_level_zero);
    std::cout << "...PASS" << std::endl;

    if (HasOpenCLDevices && !CPUs.empty()) {
      std::cout << "Test 'level_zero:gpu,cpu'";
      device d14(filter_selector("level_zero:gpu,cpu"));
      assert((d14.is_gpu() || d14.is_cpu()));
      std::cout << "...PASS 1/2" << std::endl;
      if (d14.is_gpu()) {
        assert(d14.get_platform().get_backend() ==
               backend::ext_oneapi_level_zero);
        std::cout << "...PASS 2/2" << std::endl;
      }
    }
  }

  if (Devs.size() > 1) {
    std::cout << "Test '1'";
    device d15(filter_selector("1"));
    std::cout << "...PASS" << std::endl;
  }

  if (HasCUDADevices) {
    std::cout << "Test 'cuda'";
    device d16(filter_selector("cuda"));
    assert(d16.get_platform().get_backend() == backend::ext_oneapi_cuda);
    std::cout << "...PASS" << std::endl;

    std::cout << "Test 'cuda:gpu'";
    device d17(filter_selector("cuda:gpu"));
    assert(d17.is_gpu() &&
           d17.get_platform().get_backend() == backend::ext_oneapi_cuda);
    std::cout << "...PASS" << std::endl;
  }

  if (!Accels.empty()) {
    std::cout << "Test 'accelerator'";
    device d18(filter_selector("accelerator"));
    assert(d18.is_accelerator());
    std::cout << "...PASS" << std::endl;
  }

  if (HasHIPDevices) {
    std::cout << "Test 'hip'";
    device d19(ext::oneapi::filter_selector("hip"));
    assert(d19.get_platform().get_backend() == backend::ext_oneapi_hip);
    std::cout << "...PASS" << std::endl;

    std::cout << "test 'hip:gpu'";
    device d20(ext::oneapi::filter_selector("hip:gpu"));
    assert(d20.is_gpu() &&
           d20.get_platform().get_backend() == backend::ext_oneapi_hip);
    std::cout << "...PASS" << std::endl;
  }

  return 0;
}
