//==------ filter_selector.hpp - ONEAPI filter selector -------*- C++ --*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <sycl/detail/export.hpp>             // for __SYCL_EXPORT
#include <sycl/detail/string_view.hpp>        // for string_view
#include <sycl/device.hpp>                    // for device
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
#include <sycl/device_selector.hpp> // for device_selector
#endif                              // __INTEL_PREVIEW_BREAKING_CHANGES

#include <memory> // for shared_ptr
#include <string> // for string

namespace sycl {
inline namespace _V1 {

// Forward declarations
class device;
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
class device_selector;
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
#ifdef __SYCL_INTERNAL_API
namespace ONEAPI {
class filter_selector;
}
#endif // __SYCL_INTERNAL_API

namespace ext::oneapi {
namespace detail {
class filter_selector_impl;
} // namespace detail

/// Selects a device matching one or more filters of the form
/// `Backend:DeviceType:RelativeDeviceNumber`, see
/// sycl_ext_oneapi_filter_selector.
///
/// This is a SYCL 2020 callable device selector: it can be passed to the
/// `device`, `platform` and `queue` constructors, and it may be invoked
/// directly as many times as needed. The set of devices matching the filters is
/// determined when the selector is constructed.
class __SYCL_EXPORT filter_selector
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
    // Nothing in this class needs the deprecated SYCL 1.2.1 device_selector.
    // The base class is only kept to preserve the ABI of the non-preview
    // library.
    : public device_selector
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
{
public:
  filter_selector(const std::string &filter)
      : filter_selector(sycl::detail::string_view{filter}) {}
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  int operator()(const device &dev) const override;
#else
  int operator()(const device &dev) const;
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
  /// \deprecated The selector keeps no state between the invocations of
  /// `operator()`, so there is nothing to reset.
  void reset() const;
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  device select_device() const override;
#else
  device select_device() const;
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
#ifdef __SYCL_INTERNAL_API
  friend class sycl::ONEAPI::filter_selector;
#endif

private:
  std::shared_ptr<detail::filter_selector_impl> impl;
  filter_selector(sycl::detail::string_view filter);
};
} // namespace ext::oneapi

#ifdef __SYCL_INTERNAL_API
namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
using namespace ext::oneapi;
class __SYCL_EXPORT filter_selector : public ext::oneapi::filter_selector {
public:
  filter_selector(const std::string &filter)
      : filter_selector(sycl::detail::string_view{filter}) {}
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  int operator()(const device &dev) const override;
#else
  int operator()(const device &dev) const;
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
  void reset() const;
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  device select_device() const override;
#else
  device select_device() const;
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

private:
  filter_selector(sycl::detail::string_view filter);
};
} // namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead")ONEAPI
#endif // __SYCL_INTERNAL_API
} // namespace _V1
} // namespace sycl
