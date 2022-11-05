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
  if (const char *val = std::getenv("SYCL_DEVICE_TYPE")) {
    std::string type(val);
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    if (type == "cpu") {
      return info::device_type::cpu;
    }
    if (type == "gpu") {
      return info::device_type::gpu;
    }
    if (type == "acc") {
      return info::device_type::accelerator;
    }
    if (type == "host") {
      return info::device_type::host;
    }
    throw sycl::runtime_error("SYCL_DEVICE_TYPE is not recognized.  Must "
                              "be GPU, CPU, ACC or HOST.",
                              PI_ERROR_INVALID_VALUE);
  }
  return info::device_type::all;
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
