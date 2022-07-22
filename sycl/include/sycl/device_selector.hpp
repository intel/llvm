//==------ device_selector.hpp - SYCL device selector ---------*- C++ --*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines.hpp>
#include <sycl/detail/export.hpp>

// 4.6.1 Device selection class

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// SYCL 1.2.1 defines a negative score to reject a device from selection
static constexpr int REJECT_DEVICE_SCORE = -1;

// Forward declarations
class device;

/// The SYCL 1.2.1 device_selector class provides ability to choose the
/// best SYCL device based on heuristics specified by the user.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT device_selector {

public:
  virtual ~device_selector() = default;

  virtual device select_device() const;

  virtual int operator()(const device &device) const = 0;
};

/// The default selector chooses the first available SYCL device.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT default_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

/// Selects any SYCL GPU device.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT gpu_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

/// Selects any SYCL CPU device.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT cpu_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

/// Selects any SYCL accelerator device.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT accelerator_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

/// Selects SYCL host device.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT host_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

namespace detail {

template <typename DeviceSelector,
          typename = std::enable_if_t<
              std::is_invocable_r<int, DeviceSelector &, device &>::value>>
device select_device(DeviceSelector DeviceSelectorInvocable) {
  std::vector<device> devices = device::get_devices();
  int score = REJECT_DEVICE_SCORE;
  const device *res = nullptr;

  for (const auto &dev : devices) {
    int dev_score = std::invoke(DeviceSelectorInvocable, dev);

    /*  CP - restore?
    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_ALL)) {
      std::string PlatformName = dev.get_info<info::device::platform>()
                                     .get_info<info::platform::name>();
      std::string DeviceName = dev.get_info<info::device::name>();
      std::cout << "SYCL_PI_TRACE[all]: "
                << "select_device(): -> score = " << dev_score
                << ((dev_score < 0) ? " (REJECTED)" : "") << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  platform: " << PlatformName << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  device: " << DeviceName << std::endl;
    }
    */

    // A negative score means that a device must not be selected.
    if (dev_score < 0)
      continue;

    // Section 4.6 of SYCL 1.2.1 spec: (not sure if still relevant - CP)
    // says: "If more than one device receives the high score then
    // one of those tied devices will be returned, but which of the devices
    // from the tied set is to be returned is not defined". So use the device
    // preference score to resolve ties, this is necessary for custom_selectors
    // that may not already include device preference in their scoring.

    // CP - RESTORE
    // if ((score < dev_score) || ((score == dev_score) &&
    // (getDevicePreference(*res) < getDevicePreference(dev)))) {    // <==
    // RESTORE
    if (score < dev_score) {
      res = &dev;
      score = dev_score;
    }
  }

  if (res != nullptr) {
    /* CP  - restore?
    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_BASIC)) {
      std::string PlatformName = res->get_info<info::device::platform>()
                                     .get_info<info::platform::name>();
      std::string DeviceName = res->get_info<info::device::name>();
      std::cout << "SYCL_PI_TRACE[all]: "
                << "Selected device ->" << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  platform: " << PlatformName << std::endl
                << "SYCL_PI_TRACE[all]: "
                << "  device: " << DeviceName << std::endl;
    }
    */
    return *res;
  }

  throw cl::sycl::runtime_error("No device of requested type available.",
                                PI_ERROR_DEVICE_NOT_FOUND);
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
