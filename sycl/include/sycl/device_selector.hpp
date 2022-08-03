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

// SYCL 2020 section 4.6.1.1 defines a negative score to reject a device from
// selection
static constexpr int REJECT_DEVICE_SCORE = -1;

using DSelectorInvocableType = std::function<int(const sycl::device &)>;

__SYCL_EXPORT device
select_device(const DSelectorInvocableType &DeviceSelectorInvocable);

__SYCL_EXPORT device
select_device(const DSelectorInvocableType &DeviceSelectorInvocable,
              const context &SyclContext);

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
