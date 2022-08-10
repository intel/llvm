//==------ device_selector.cpp - SYCL device selector ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>
#include <detail/device_impl.hpp>
#include <detail/filter_selector_impl.hpp>
#include <detail/force_device.hpp>
#include <detail/global_handler.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/detail/device_filter.hpp>
#include <sycl/device.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/filter_selector.hpp>
#include <sycl/stl.hpp>
// 4.6.1 Device selection class

#include <algorithm>
#include <cctype>
#include <regex>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

namespace detail {

// SYCL_DEVICE_FILTER doesn't need to be considered in the device preferences
// as it filters the device list returned by device::get_devices itself, so
// only matching devices will be scored.
static int getDevicePreference(const device &Device) {
  int Score = 0;

  // No preferences for host devices.
  if (Device.is_host())
    return Score;

  // Strongly prefer devices with available images.
  auto &program_manager = sycl::detail::ProgramManager::getInstance();
  if (program_manager.hasCompatibleImage(Device))
    Score += 1000;

  // Prefer level_zero backend devices.
  if (detail::getSyclObjImpl(Device)->getPlugin().getBackend() ==
      backend::ext_oneapi_level_zero)
    Score += 50;

  return Score;
}

static void traceDeviceSelection(const device &Device, int Score, bool Chosen) {
  bool shouldTrace = false;
  if (Chosen) {
    shouldTrace = detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_BASIC);
  } else {
    shouldTrace = detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_ALL);
  }
  if (shouldTrace) {
    std::string PlatformName = Device.get_info<info::device::platform>()
                                   .get_info<info::platform::name>();
    std::string DeviceName = Device.get_info<info::device::name>();
    auto selectionMsg = Chosen ? "Selected device: -> final score = "
                               : "Candidate device: -> score = ";

    std::cout << "SYCL_PI_TRACE[all]: " << selectionMsg << Score
              << ((Score < 0) ? " (REJECTED)" : "") << std::endl
              << "SYCL_PI_TRACE[all]: "
              << "  platform: " << PlatformName << std::endl
              << "SYCL_PI_TRACE[all]: "
              << "  device: " << DeviceName << std::endl;
  }
}

device select_device(DSelectorInvocableType DeviceSelectorInvocable,
                     std::vector<device> &Devices) {
  int score = detail::REJECT_DEVICE_SCORE;
  const device *res = nullptr;

  for (const auto &dev : Devices) {
    int dev_score = DeviceSelectorInvocable(dev);

    traceDeviceSelection(dev, dev_score, false);

    // A negative score means that a device must not be selected.
    if (dev_score < 0)
      continue;

    // Section 4.6 of SYCL 1.2.1 spec:
    // "If more than one device receives the high score then
    // one of those tied devices will be returned, but which of the devices
    // from the tied set is to be returned is not defined". So use the device
    // preference score to resolve ties, this is necessary for custom_selectors
    // that may not already include device preference in their scoring.

    if ((score < dev_score) ||
        ((score == dev_score) &&
         (getDevicePreference(*res) < getDevicePreference(dev)))) {
      res = &dev;
      score = dev_score;
    }
  }

  if (res != nullptr) {
    traceDeviceSelection(*res, score, true);

    return *res;
  }

  throw sycl::runtime_error("No device of requested type available.",
                            PI_ERROR_DEVICE_NOT_FOUND);
}

// select_device(selector)
__SYCL_EXPORT device
select_device(const DSelectorInvocableType &DeviceSelectorInvocable) {
  std::vector<device> Devices = device::get_devices();

  return select_device(DeviceSelectorInvocable, Devices);
}

// select_device(selector, context)
__SYCL_EXPORT device
select_device(const DSelectorInvocableType &DeviceSelectorInvocable,
              const context &SyclContext) {
  std::vector<device> devices = SyclContext.get_devices();

  return select_device(DeviceSelectorInvocable, devices);
}

} // namespace detail

device device_selector::select_device() const {
  return detail::select_device([&](const device &dev) { return (*this)(dev); });
}

/// Devices of different kinds are prioritized in the following order:
/// 1. GPU
/// 2. CPU
/// 3. Host
/// 4. Accelerator
int default_selector::operator()(const device &dev) const {
  // The default selector doesn't reject any devices.
  int Score = 0;

  if (dev.get_info<info::device::device_type>() == detail::get_forced_type())
    Score += 2000;

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

  // Add preference score.
  Score += detail::getDevicePreference(dev);

  return Score;
}

int gpu_selector::operator()(const device &dev) const {
  int Score = detail::REJECT_DEVICE_SCORE;

  if (dev.is_gpu()) {
    Score = 1000;
    Score += detail::getDevicePreference(dev);
  }
  return Score;
}

int cpu_selector::operator()(const device &dev) const {
  int Score = detail::REJECT_DEVICE_SCORE;

  if (dev.is_cpu()) {
    Score = 1000;
    Score += detail::getDevicePreference(dev);
  }
  return Score;
}

int accelerator_selector::operator()(const device &dev) const {
  int Score = detail::REJECT_DEVICE_SCORE;

  if (dev.is_accelerator()) {
    Score = 1000;
    Score += detail::getDevicePreference(dev);
  }
  return Score;
}

int host_selector::operator()(const device &dev) const {
  int Score = detail::REJECT_DEVICE_SCORE;

  if (dev.is_host()) {
    Score = 1000;
    Score += detail::getDevicePreference(dev);
  }
  return Score;
}

namespace ext {
namespace oneapi {

filter_selector::filter_selector(const std::string &Input)
    : impl(std::make_shared<detail::filter_selector_impl>(Input)) {}

int filter_selector::operator()(const device &Dev) const {
  return impl->operator()(Dev);
}

void filter_selector::reset() const { impl->reset(); }

// filter_selectors not "Callable"
// because of the requirement that the filter_selector "reset()" itself
// between invocations, the filter_selector operator() is not purely callable
// and cannot be used interchangeably as a SYCL2020 callable device selector.
// TODO: replace the FilterSelector subclass with something that
// doesn't pretend to be a device_selector, and instead is something that
// just returns a device (rather than a score).
// Then remove ! std::is_base_of_v<ext::oneapi::filter_selector, DeviceSelector>
// from device/platform/queue constructors
device filter_selector::select_device() const {
  std::lock_guard<std::mutex> Guard(
      sycl::detail::GlobalHandler::instance().getFilterMutex());

  device Result = device_selector::select_device();

  reset();

  return Result;
}

} // namespace oneapi
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
  using namespace ext::oneapi;
  filter_selector::filter_selector(const std::string &Input)
      : ext::oneapi::filter_selector(Input) {}

  int filter_selector::operator()(const device &Dev) const {
    return ext::oneapi::filter_selector::operator()(Dev);
  }

  void filter_selector::reset() const { ext::oneapi::filter_selector::reset(); }

  device filter_selector::select_device() const {
    return ext::oneapi::filter_selector::select_device();
  }
} // namespace ONEAPI

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
