//==-- filter_selector_impl.hpp - oneapi filter selector impl--*- C++ --*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/device_filter.hpp>
#include <sycl/device_selector.hpp>

#include <vector>

namespace sycl {
inline namespace _V1 {

// Forward declarations
class device;

namespace ext {
namespace oneapi {
namespace detail {

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
typedef struct sycl::detail::ods_target filter;
#else
typedef struct sycl::detail::device_filter filter;
#endif

class filter_selector_impl {
public:
  filter_selector_impl(const std::string &filter);
  int operator()(const device &dev) const;
  void reset() const;

private:
  static constexpr int REJECT_DEVICE_SCORE = -1;
  mutable std::vector<filter> mFilters;
  mutable int mNumDevicesSeen;
  int mNumTotalDevices;
  mutable bool mMatchFound;
};
} // namespace detail
} // namespace oneapi
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
using namespace ext::oneapi;
}
} // namespace _V1
} // namespace sycl
