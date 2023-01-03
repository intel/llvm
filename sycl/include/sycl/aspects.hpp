//==--------------- aspects.hpp - SYCL Aspect Enums ------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <sycl/detail/defines.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

#define __SYCL_ASPECT(ASPECT, ID) ASPECT = ID,
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE)                          \
  ASPECT __SYCL2020_DEPRECATED(MESSAGE) = ID,
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)                    \
  __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE)
enum class __SYCL_TYPE(aspect) aspect {
#include <sycl/info/aspects.def>
#include <sycl/info/aspects_deprecated.def>
};
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
