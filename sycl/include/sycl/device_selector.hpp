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

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

// Forward declarations
class device;

namespace ext {
namespace oneapi {
class filter_selector;
}
} // namespace ext

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

// -------------- SYCL 2020

// SYCL 2020 standalone selectors
__SYCL_EXPORT int default_selector_v(const device &dev);
__SYCL_EXPORT int gpu_selector_v(const device &dev);
__SYCL_EXPORT int cpu_selector_v(const device &dev);
__SYCL_EXPORT int accelerator_selector_v(const device &dev);

namespace detail {
// SYCL 2020 section 4.6.1.1 defines a negative score to reject a device from
// selection
static constexpr int REJECT_DEVICE_SCORE = -1;

using DSelectorInvocableType = std::function<int(const sycl::device &)>;

#if __cplusplus >= 201703L

// Enable if DeviceSelector callable has matching signature, but
// exclude if descended from filter_selector which is not purely callable.
// See [FilterSelector not Callable] in device_selector.cpp
template <typename DeviceSelector>
using EnableIfDeviceSelectorInvocable = std::enable_if_t<
    std::is_invocable_r_v<int, DeviceSelector &, const device &> &&
    !std::is_base_of_v<ext::oneapi::filter_selector, DeviceSelector>>;
#endif

__SYCL_EXPORT device
select_device(const DSelectorInvocableType &DeviceSelectorInvocable);

__SYCL_EXPORT device
select_device(const DSelectorInvocableType &DeviceSelectorInvocable,
              const context &SyclContext);
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
