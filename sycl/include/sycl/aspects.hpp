//==--------------- aspects.hpp - SYCL Aspect Enums ------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <sycl/detail/defines.hpp>            // for __SYCL_TYPE
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

namespace sycl {
inline namespace _V1 {

#define __SYCL_ASPECT(ASPECT, ID) ASPECT = ID,
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE)                          \
  ASPECT __SYCL2020_DEPRECATED(MESSAGE) = ID,
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

enum class __SYCL_TYPE(aspect) aspect {
#include <sycl/info/aspects.def>
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
#include <sycl/info/aspects_deprecated.def>
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
};

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
#undef __SYCL_ASPECT_DEPRECATED
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
#undef __SYCL_ASPECT

} // namespace _V1
} // namespace sycl
