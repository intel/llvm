//==---------- force_device.hpp - Forcing SYCL device ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/detail/defines.hpp>
#include <sycl/__impl/info/info_desc.hpp>

#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
#else
namespace __sycl_internal {
inline namespace __v1 {
#endif
namespace detail {

bool match_types(const info::device_type &l, const info::device_type &r);

info::device_type get_forced_type();

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
