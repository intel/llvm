//==-- filter_selector_impl.hpp - oneapi filter selector impl--*- C++ --*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/device_filter.hpp>
#include <sycl/device.hpp>

#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {

namespace ext {
namespace oneapi {
namespace detail {

using filter = sycl::detail::ods_target;

/// The set of devices matching the filter string is computed once, when the
/// selector is created. That keeps operator() a pure function, so that the
/// selector can be used as a SYCL 2020 callable device selector.
class filter_selector_impl {
public:
  filter_selector_impl(const std::string &filter);
  int operator()(const device &dev) const;

private:
  static constexpr int REJECT_DEVICE_SCORE = -1;
  std::vector<device> mMatchingDevices;
};
} // namespace detail
} // namespace oneapi
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
using namespace ext::oneapi;
}
} // namespace _V1
} // namespace sycl
