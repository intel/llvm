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

namespace cl {
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

#ifndef SYCL_ERR_MSG_DEVICE_IS_NOT_AVAILABLE
#define SYCL_ERR_MSG_DEVICE_IS_NOT_AVAILABLE                                   \
  "No device of the requested type available."
#endif // SYCL_ERR_MSG_DEVICE_IS_NOT_AVAILABLE
  throw cl::sycl::runtime_error(SYCL_ERR_MSG_DEVICE_IS_NOT_AVAILABLE);
}

int default_selector::operator()(const device &dev) const {
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
} // namespace cl
