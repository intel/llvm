//==----------- sub_group.hpp --- SYCL sub-group ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/group.hpp>
#include <sycl/sub_group.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi {
struct __SYCL_DEPRECATED(
    "use sycl::ext::oneapi::experimental::this_sub_group() instead") sub_group
    : sycl::sub_group {
  // These two constructors are intended to keep the correctness of such code
  // after the sub_group class migration from ext::oneapi to the sycl namespace:
  // sycl::ext::oneapi::sub_group sg =
  //    sycl::ext::oneapi::experimental::this_sub_group();
  // ...
  // sycl::ext::oneapi::sub_group sg = item.get_sub_group();
  sub_group(const sycl::sub_group &sg) {
#ifdef __SYCL_DEVICE_ONLY__
    sub_group();
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

private:
  sub_group() = default;
};
} // namespace ext::oneapi
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
