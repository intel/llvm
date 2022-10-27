//==---------- force_device.cpp - Forcing SYCL device ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/force_device.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/stl.hpp>

#include <algorithm>
#include <cstdlib>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

bool match_types(const info::device_type &l, const info::device_type &r) {
  return l == info::device_type::all || l == r || r == info::device_type::all;
}

info::device_type get_forced_type() {
  return info::device_type::all;
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
