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
#include <detail/force_device.hpp>
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

  throw cl::sycl::runtime_error("No device of requested type available.",
                                PI_DEVICE_NOT_FOUND);
}

/// Devices of different kinds are prioritized in the following order:
/// 1. GPU
/// 2. Accelerator
/// 3. CPU
/// 4. Host
int default_selector::operator()(const device &dev) const {

  // Take note of the SYCL_BE environment variable when doing default selection
  const char *SYCL_BE = std::getenv("SYCL_BE");
  if (SYCL_BE) {
    std::string backend = (SYCL_BE ? SYCL_BE : "");
    // Taking the version information from the platform gives us more useful
    // information than the driver_version of the device.
    const platform platform = dev.get_info<info::device::platform>();
    const std::string platformVersion =
        platform.get_info<info::platform::version>();
    ;
    // If using PI_CUDA, don't accept a non-CUDA device
    if (platformVersion.find("CUDA") == std::string::npos &&
        backend == "PI_CUDA") {
      return -1;
    }
    // If using PI_OPENCL, don't accept a non-OpenCL device
    if (platformVersion.find("OpenCL") == std::string::npos &&
        backend == "PI_OPENCL") {
      return -1;
    }
  }

  // override always wins
  if (dev.get_info<info::device::device_type>() == detail::get_forced_type())
    return 1000;

  if (dev.is_gpu())
    return 500;

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
