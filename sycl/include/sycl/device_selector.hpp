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
__SYCL_EXPORT int getDevicePreference(const device &Device);
__SYCL_EXPORT void traceDeviceSelection(const device &Device, int score,
                                        bool chosen);
// returns the range of devices against which we will be running the device selector.
__SYCL_EXPORT std::vector<device> getDevices();
__SYCL_EXPORT std::vector<device> getDevices(const context &SyclContext);

#if __cplusplus >= 201703L
// select_device(selector, deviceVector)
template <typename DeviceSelector,
          typename = std::enable_if_t<
              std::is_invocable_r<int, DeviceSelector &, device &>::value>>
device select_device(DeviceSelector DeviceSelectorInvocable,
                     std::vector<device> &devices) {
  int score = REJECT_DEVICE_SCORE;
  const device *res = nullptr;

  for (const auto &dev : devices) {
    int dev_score = std::invoke(DeviceSelectorInvocable, dev);

    traceDeviceSelection(dev, dev_score, false);

    // A negative score means that a device must not be selected.
    if (dev_score < 0)
      continue;

    // Section 4.6 of SYCL 1.2.1 spec:
    // says: "If more than one device receives the high score then
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

  throw cl::sycl::runtime_error("No device of requested type available.",
                                PI_ERROR_DEVICE_NOT_FOUND);
}

// select_device(selector)
template <typename DeviceSelector,
          typename = std::enable_if_t<
              std::is_invocable_r<int, DeviceSelector &, device &>::value>>
device select_device(DeviceSelector DeviceSelectorInvocable) {
  std::vector<device> devices = getDevices();

  return select_device(DeviceSelectorInvocable, devices);
}

// select_device(selector, context)
template <typename DeviceSelector,
          typename = std::enable_if_t<
              std::is_invocable_r<int, DeviceSelector &, device &>::value>>
device select_device(DeviceSelector DeviceSelectorInvocable,
                     const context &SyclContext) {
  std::vector<device> devices = getDevices(SyclContext);

  return select_device(DeviceSelectorInvocable, devices);
}
#endif
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
