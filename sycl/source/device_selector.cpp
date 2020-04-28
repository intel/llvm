//==------ device_selector.cpp - SYCL device selector ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/config.hpp>
#include <detail/device_impl.hpp>
#include <detail/force_device.hpp>
// 4.6.1 Device selection class

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Utility function to check if device is of the preferred backend.
// Currently preference is given to the opencl backend.
static bool isDeviceOfPreferredSyclBe(const device &Device) {
  if (Device.is_host())
    return false;

  return detail::getSyclObjImpl(Device)->getPlugin().getBackend() ==
         backend::opencl;
}

// @return True if the device is invalid for the current backend preferences
static bool isDeviceInvalidForBe(const device &Device) {

  if (Device.is_host())
    return false;

  // Taking the version information from the platform gives us more useful
  // information than the driver_version of the device.
  const platform platform = Device.get_info<info::device::platform>();
  const std::string platformVersion =
      platform.get_info<info::platform::version>();

  backend *BackendPref = detail::SYCLConfig<detail::SYCL_BE>::get();
  auto BackendType = detail::getSyclObjImpl(Device)->getPlugin().getBackend();
  static_assert(std::is_same<backend, decltype(BackendType)>(),
                "Type is not the same");

  // If no preference, assume OpenCL and reject CUDA backend
  if (BackendType == backend::cuda && !BackendPref) {
    return true;
  } else if (!BackendPref)
    return false;

  // If using PI_CUDA, don't accept a non-CUDA device
  if (BackendType == backend::opencl && *BackendPref == backend::cuda)
    return true;

  // If using PI_OPENCL, don't accept a non-OpenCL device
  if (BackendType == backend::cuda && *BackendPref == backend::opencl)
    return true;

  return false;
}

device device_selector::select_device() const {
  vector_class<device> devices = device::get_devices();
  int score = -1;
  const device *res = nullptr;
  for (const auto &dev : devices) {

    // Reject the NVIDIA OpenCL platform
    if (!dev.is_host()) {
      string_class PlatformName = dev.get_info<info::device::platform>()
                                      .get_info<info::platform::name>();
      const bool IsCUDAPlatform =
          PlatformName.find("CUDA") != std::string::npos;

      if (detail::getSyclObjImpl(dev)->getPlugin().getBackend() ==
              backend::opencl &&
          IsCUDAPlatform) {
        continue;
      }
    }

    int dev_score = (*this)(dev);
    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_ALL)) {
      string_class PlatformVersion = dev.get_info<info::device::platform>()
                                         .get_info<info::platform::version>();
      string_class DeviceName = dev.get_info<info::device::name>();
      std::cout << "SYCL_PI_TRACE[all]: "
                << "select_device(): -> score = " << score << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  platform: " << PlatformVersion << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  device: " << DeviceName << std::endl;
    }

    // SYCL spec says: "If more than one device receives the high score then
    // one of those tied devices will be returned, but which of the devices
    // from the tied set is to be returned is not defined". Here we give a
    // preference to the device of the preferred BE.
    //
    if (score < dev_score ||
        (score == dev_score && isDeviceOfPreferredSyclBe(dev))) {
      res = &dev;
      score = dev_score;
    }
  }

  if (res != nullptr) {
    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_BASIC)) {
      string_class PlatformVersion = res->get_info<info::device::platform>()
                                         .get_info<info::platform::version>();
      string_class DeviceName = res->get_info<info::device::name>();
      std::cout << "SYCL_PI_TRACE[all]: "
                << "Selected device ->" << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  platform: " << PlatformVersion << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  device: " << DeviceName << std::endl;
    }
    return *res;
  }

  throw cl::sycl::runtime_error("No device of requested type available.",
                                PI_DEVICE_NOT_FOUND);
}

int default_selector::operator()(const device &dev) const {

  int Score = -1;

  if (isDeviceInvalidForBe(dev))
    return -1;

  // Give preference to device of SYCL BE.
  if (isDeviceOfPreferredSyclBe(dev))
    Score = 50;

  // override always wins
  if (dev.get_info<info::device::device_type>() == detail::get_forced_type())
    Score += 1000;

  if (dev.is_gpu())
    Score += 500;

  if (dev.is_cpu())
    Score += 300;

  if (dev.is_host())
    Score += 100;

  return Score;
}

int gpu_selector::operator()(const device &dev) const {
  int Score = -1;

  if (isDeviceInvalidForBe(dev))
    return -1;

  if (dev.is_gpu()) {
    Score = 1000;
    // Give preference to device of SYCL BE.
    if (isDeviceOfPreferredSyclBe(dev))
      Score = 50;
  }
  return Score;
}

int cpu_selector::operator()(const device &dev) const {
  int Score = -1;

  if (isDeviceInvalidForBe(dev))
    return -1;

  if (dev.is_cpu()) {
    Score = 1000;

    // Give preference to device of SYCL BE.
    if (isDeviceOfPreferredSyclBe(dev))
      Score += 50;
  }
  return Score;
}

int accelerator_selector::operator()(const device &dev) const {
  int Score = -1;

  if (isDeviceInvalidForBe(dev))
    return -1;

  if (dev.is_accelerator()) {
    Score = 1000;
    // Give preference to device of SYCL BE.
    if (isDeviceOfPreferredSyclBe(dev))
      Score += 50;
  }
  return Score;
}

int host_selector::operator()(const device &dev) const {
  int Score = -1;
  if (dev.is_host()) {
    Score = 1000;
    // Give preference to device of SYCL BE.
    if (isDeviceOfPreferredSyclBe(dev)) {
      Score += 50;
    } else if (isDeviceInvalidForBe(dev))
      return -1;
  }
  return Score;
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
