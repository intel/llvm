//==------ device_selector.cpp - SYCL device selector ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/ONEAPI/filter_selector.hpp>
#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/device_filter.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/config.hpp>
#include <detail/device_impl.hpp>
#include <detail/filter_selector_impl.hpp>
#include <detail/force_device.hpp>
#include <detail/global_handler.hpp>
// 4.6.1 Device selection class

#include <algorithm>
#include <cctype>
#include <regex>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Utility function to check if device is of the preferred backend.
// Currently preference is given to the level_zero backend.
static bool isDeviceOfPreferredSyclBe(const device &Device) {
  if (Device.is_host())
    return false;

  return detail::getSyclObjImpl(Device)->getPlugin().getBackend() ==
         backend::level_zero;
}

// Return true if the given device 'Dev' matches with any filter
static bool isForcedDevice(const device &Dev, int Index = -1) {
  detail::device_filter_list *FilterList =
      detail::SYCLConfig<detail::SYCL_DEVICE_FILTER>::get();

  if (!FilterList)
    return false;
  info::device_type Type = Dev.get_info<info::device::device_type>();
  backend Backend;
  if (Type == info::device_type::host)
    Backend = backend::host;
  else
    Backend = detail::getSyclObjImpl(Dev)->getPlugin().getBackend();

  for (const detail::device_filter &Filter : FilterList->get()) {
    if ((Filter.Backend == Backend || Filter.Backend == backend::all) &&
        (Filter.DeviceType == Type ||
         Filter.DeviceType == info::device_type::all)) {
      if (Index < 0 || (Filter.HasDeviceNum && Filter.DeviceNum == Index))
        return true;
    }
  }
  return false;
}

device device_selector::select_device() const {
  vector_class<device> devices = device::get_devices();
  int score = REJECT_DEVICE_SCORE;
  const device *res = nullptr;

  for (const auto &dev : devices) {
    int dev_score = (*this)(dev);

    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_ALL)) {
      string_class PlatformName = dev.get_info<info::device::platform>()
                                      .get_info<info::platform::name>();
      string_class DeviceName = dev.get_info<info::device::name>();
      std::cout << "SYCL_PI_TRACE[all]: "
                << "select_device(): -> score = " << dev_score
                << ((dev_score < 0) ? " (REJECTED)" : "") << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  platform: " << PlatformName << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  device: " << DeviceName << std::endl;
    }

    // A negative score means that a device must not be selected.
    if (dev_score < 0)
      continue;

    // If SYCL_DEVICE_FILTER is set, give a bonus point for the device
    // whose index matches with desired device number.
    int index = &dev - &devices[0];
    if (isForcedDevice(dev, index)) {
      dev_score += 1000;
    }

    // SYCL spec says: "If more than one device receives the high score then
    // one of those tied devices will be returned, but which of the devices
    // from the tied set is to be returned is not defined". Here we give a
    // preference to the device of the preferred BE.
    //
    if ((score < dev_score) ||
        (score == dev_score && isDeviceOfPreferredSyclBe(dev))) {
      res = &dev;
      score = dev_score;
    }
  }

  if (res != nullptr) {
    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_BASIC)) {
      string_class PlatformName = res->get_info<info::device::platform>()
                                      .get_info<info::platform::name>();
      string_class DeviceName = res->get_info<info::device::name>();
      std::cout << "SYCL_PI_TRACE[all]: "
                << "Selected device ->" << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  platform: " << PlatformName << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  device: " << DeviceName << std::endl;
    }
    return *res;
  }

  throw cl::sycl::runtime_error("No device of requested type available.",
                                PI_DEVICE_NOT_FOUND);
}

/// Devices of different kinds are prioritized in the following order:
/// 1. GPU
/// 2. CPU
/// 3. Host
int default_selector::operator()(const device &dev) const {

  int Score = REJECT_DEVICE_SCORE;

  // Give preference to device of SYCL BE.
  if (isDeviceOfPreferredSyclBe(dev))
    Score = 50;

  // If SYCL_DEVICE_FILTER is set, filter device gets a high point.
  // All unmatched devices should never be selected.
  detail::device_filter_list *FilterList =
      detail::SYCLConfig<detail::SYCL_DEVICE_FILTER>::get();
  if (FilterList) {
    if (isForcedDevice(dev))
      Score = 1000;
    else
      return REJECT_DEVICE_SCORE;
  }

  if (dev.get_info<info::device::device_type>() == detail::get_forced_type())
    Score += 1000;

  if (dev.is_gpu())
    Score += 500;

  if (dev.is_cpu())
    Score += 300;

  if (dev.is_host())
    Score += 100;

  // Since we deprecate SYCL_BE and SYCL_DEVICE_TYPE,
  // we should not disallow accelerator to be chosen.
  // But this device type gets the lowest heuristic point.
  if (dev.is_accelerator())
    Score += 75;

  return Score;
}

int gpu_selector::operator()(const device &dev) const {
  int Score = REJECT_DEVICE_SCORE;

  if (dev.is_gpu()) {
    detail::device_filter_list *FilterList =
        detail::SYCLConfig<detail::SYCL_DEVICE_FILTER>::get();
    if (FilterList) {
      if (isForcedDevice(dev))
        Score = 1000;
      else
        return Score;
    } else {
      Score = 1000;
    }
    // Give preference to device of SYCL BE.
    if (isDeviceOfPreferredSyclBe(dev))
      Score += 50;
  }
  return Score;
}

int cpu_selector::operator()(const device &dev) const {
  int Score = REJECT_DEVICE_SCORE;

  if (dev.is_cpu()) {
    detail::device_filter_list *FilterList =
        detail::SYCLConfig<detail::SYCL_DEVICE_FILTER>::get();
    if (FilterList) {
      if (isForcedDevice(dev))
        Score = 1000;
      else
        return Score;
    } else {
      Score = 1000;
    }
    // Give preference to device of SYCL BE.
    if (isDeviceOfPreferredSyclBe(dev))
      Score += 50;
  }
  return Score;
}

int accelerator_selector::operator()(const device &dev) const {
  int Score = REJECT_DEVICE_SCORE;

  if (dev.is_accelerator()) {
    detail::device_filter_list *FilterList =
        detail::SYCLConfig<detail::SYCL_DEVICE_FILTER>::get();
    if (FilterList) {
      if (isForcedDevice(dev))
        Score = 1000;
      else
        return Score;
    } else {
      Score = 1000;
    }
    // Give preference to device of SYCL BE.
    if (isDeviceOfPreferredSyclBe(dev))
      Score += 50;
  }
  return Score;
}

int host_selector::operator()(const device &dev) const {
  int Score = REJECT_DEVICE_SCORE;

  if (dev.is_host()) {
    Score = 1000;
    // Give preference to device of SYCL BE.
    if (isDeviceOfPreferredSyclBe(dev))
      Score += 50;
  }
  return Score;
}

namespace ONEAPI {

filter_selector::filter_selector(const std::string &Input)
    : impl(std::make_shared<detail::filter_selector_impl>(Input)) {}

int filter_selector::operator()(const device &Dev) const {
  return impl->operator()(Dev);
}

void filter_selector::reset() const { impl->reset(); }

device filter_selector::select_device() const {
  std::lock_guard<std::mutex> Guard(
      sycl::detail::GlobalHandler::instance().getFilterMutex());

  device Result = device_selector::select_device();

  reset();

  return Result;
}

} // namespace ONEAPI
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
