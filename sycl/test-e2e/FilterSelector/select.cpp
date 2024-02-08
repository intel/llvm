// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out

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

  CPUs = device::get_devices(sycl::info::device_type::cpu);
  GPUs = device::get_devices(sycl::info::device_type::gpu);
  Accels = device::get_devices(sycl::info::device_type::accelerator);
  Devs = device::get_devices();

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

  if (!CPUs.empty()) {
    device d1(filter_selector("cpu"));
    assert(d1.is_cpu() && "filter_selector(\"cpu\") failed");
  }

  if (!GPUs.empty()) {
    device d2(filter_selector("gpu"));
    assert(d2.is_gpu() && "filter_selector(\"gpu\") failed");
  }

  if (!CPUs.empty() || !GPUs.empty()) {
    device d3(filter_selector("cpu,gpu"));
    assert((d3.is_gpu() || d3.is_cpu()) &&
           "filter_selector(\"cpu,gpu\") failed");
  }

  if (HasOpenCLDevices) {
    device d4(filter_selector("opencl"));
    assert(d4.get_platform().get_backend() == backend::opencl &&
           "filter_selector(\"opencl\") failed");

    if (!CPUs.empty()) {
      device d5(filter_selector("opencl:cpu"));
      assert(d5.is_cpu() &&
             d5.get_platform().get_backend() == backend::opencl &&
             "filter_selector(\"opencl:cpu\") failed");

      device d6(filter_selector("opencl:cpu:0"));
      assert(d6.is_cpu() &&
             d6.get_platform().get_backend() == backend::opencl &&
             "filter_selector(\"opencl:cpu:0\") failed");
    }

    if (HasOpenCLGPU) {
      device d7(filter_selector("opencl:gpu"));
      assert(d7.is_gpu() &&
             d7.get_platform().get_backend() == backend::opencl &&
             "filter_selector(\"opencl:gpu\") failed");
    }
  }

  device d8(filter_selector("0"));

  try {
    // pick something crazy
    device d9(filter_selector("gpu:999"));
  } catch (const sycl::runtime_error &e) {
    const char *ErrorMesg =
        "Could not find a device that matches the specified filter(s)!";
    assert(std::string{e.what()}.find(ErrorMesg) == 0 &&
           "filter_selector(\"gpu:999\") unexpectedly selected a device");
  }

  try {
    // pick something crazy
    device d10(filter_selector("bob:gpu"));
  } catch (const sycl::runtime_error &e) {
    const char *ErrorMesg = "Invalid filter string!";
    assert(std::string{e.what()}.find(ErrorMesg) == 0 &&
           "filter_selector(\"bob:gpu\") unexpectedly selected a device");
  }

  try {
    // pick something crazy
    device d11(filter_selector("opencl:bob"));
  } catch (const sycl::runtime_error &e) {
    const char *ErrorMesg = "Invalid filter string!";
    assert(std::string{e.what()}.find(ErrorMesg) == 0 &&
           "filter_selector(\"opencl:bob\") unexpectedly selected a device");
  }

  if (HasLevelZeroDevices && HasLevelZeroGPU) {
    device d12(filter_selector("level_zero"));
    assert(d12.get_platform().get_backend() == backend::ext_oneapi_level_zero &&
           "filter_selector(\"level_zero\") failed");

    device d13(filter_selector("level_zero:gpu"));
    assert(d13.is_gpu() &&
           d13.get_platform().get_backend() == backend::ext_oneapi_level_zero &&
           "filter_selector(\"level_zero:gpu\") failed");

    if (HasOpenCLDevices && !CPUs.empty()) {
      device d14(filter_selector("level_zero:gpu,cpu"));
      assert((d14.is_gpu() || d14.is_cpu()) &&
             "filter_selector(\"level_zero:gpu,cpu\") failed");
      if (d14.is_gpu()) {
        assert(d14.get_platform().get_backend() ==
                   backend::ext_oneapi_level_zero &&
               "filter_selector(\"level_zero:gpu,cpu\") failed");
      }
    }
  }

  if (Devs.size() > 1) {
    device d15(filter_selector("1"));
  }

  if (HasCUDADevices) {
    device d16(filter_selector("cuda"));
    assert(d16.get_platform().get_backend() == backend::ext_oneapi_cuda &&
           "filter_selector(\"cuda\") failed");

    device d17(filter_selector("cuda:gpu"));
    assert(d17.is_gpu() &&
           d17.get_platform().get_backend() == backend::ext_oneapi_cuda &&
           "filter_selector(\"cuda:gpu\") failed");
  }

  if (!Accels.empty()) {
    device d18(filter_selector("accelerator"));
    assert(d18.is_accelerator() && "filter_selector(\"accelerator\") failed");
  }

  if (HasHIPDevices) {
    device d19(ext::oneapi::filter_selector("hip"));
    assert(d19.get_platform().get_backend() == backend::ext_oneapi_hip &&
           "filter_selector(\"hip\") failed");

    device d20(ext::oneapi::filter_selector("hip:gpu"));
    assert(d20.is_gpu() &&
           d20.get_platform().get_backend() == backend::ext_oneapi_hip &&
           "filter_selector(\"hip:gpu\") failed");
  }

  return 0;
}
