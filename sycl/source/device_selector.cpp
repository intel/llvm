//==------ device_selector.cpp - SYCL device selector ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/stl.hpp>
// 4.6.1 Device selection class

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
device device_selector::select_device() const {
  vector_class<device> devices = device::get_devices();
  int score = -1;
  const device *res = nullptr;
  for (const auto &dev : devices)
    if (score < operator()(dev)) {
      res = &dev;
      score = operator()(dev);
    }

  if (res != nullptr)
    return *res;

  throw cl::sycl::runtime_error("No device of requested type available.");
}

int default_selector::operator()(const device &dev) const {

  // Take note of the SYCL_BE environment variable when doing default selection
  const char *SYCL_BE = std::getenv("SYCL_BE");
  std::string backend = (SYCL_BE ? SYCL_BE : "");
  if (backend != "") {
    const std::string DriverVersion = dev.get_info<info::device::driver_version>();
    // If we have a cuda device but aren't using PI_CUDA, don't use this device
    if (DriverVersion.find("CUDA") == std::string::npos && backend == "PI_CUDA") {
      return -1;
    }
    // We can't easily check for an OpenCL device as there is no common string
    // to search for on all devices, so just guarantee we don't choose a cuda
    // device when PI_OPENCL is used
    if (DriverVersion.find("CUDA") != std::string::npos && backend == "PI_OPENCL") {
      return -1;
    }
  }

  if (dev.is_gpu())
    return 500;

  if (dev.is_accelerator())
    return 400;

  if (dev.is_cpu())
    return 300;

  if (dev.is_host())
    return 100;

  return -1;
}

int gpu_selector::operator()(const device &dev) const {
  return dev.is_gpu() ? 1000 : -1;
}

int cpu_selector::operator()(const device &dev) const {
  return dev.is_cpu() ? 1000 : -1;
}

int accelerator_selector::operator()(const device &dev) const {
  return dev.is_accelerator() ? 1000 : -1;
}

int host_selector::operator()(const device &dev) const {
  return dev.is_host() ? 1000 : -1;
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
