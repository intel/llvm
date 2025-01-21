//==------ filter_selector.hpp - ONEAPI filter selector -------*- C++ --*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/export.hpp>   // for __SYCL_EXPORT
#include <sycl/device.hpp>          // for device
#include <sycl/device_selector.hpp> // for device_selector

#include <memory> // for shared_ptr
#include <string> // for string

// 4.6.1 Device selection class

namespace sycl {
inline namespace _V1 {

// Forward declarations
class device;
class device_selector;
#ifdef __SYCL_INTERNAL_API
namespace ONEAPI {
class filter_selector;
}
#endif // __SYCL_INTERNAL_API

namespace ext::oneapi {
namespace detail {
class filter_selector_impl;
} // namespace detail

class __SYCL_EXPORT filter_selector : public device_selector {
public:
  filter_selector(const std::string &filter)
      : filter_selector(sycl::detail::string_view{filter}) {}
  int operator()(const device &dev) const override;
  void reset() const;
  device select_device() const override;
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
  int operator()(const device &dev) const override;
  void reset() const;
  device select_device() const override;

private:
  filter_selector(sycl::detail::string_view filter);
};
} // namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead")ONEAPI
#endif // __SYCL_INTERNAL_API
} // namespace _V1
} // namespace sycl
