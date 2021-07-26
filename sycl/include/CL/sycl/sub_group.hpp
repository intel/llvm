//==----------- sub_group.hpp --- SYCL sub-group ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/group.hpp>
#include <sycl/ext/oneapi/sub_group.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
using ext::oneapi::sub_group;
// TODO move the entire sub_group class implementation to this file once
// breaking changes are allowed.

namespace ext {
namespace oneapi {
namespace experimental {
inline sub_group this_sub_group() {
#ifdef __SYCL_DEVICE_ONLY__
  return sub_group();
#else
  throw runtime_error("Sub-groups are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
