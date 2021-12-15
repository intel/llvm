//===--- level_zero_ownership.hpp - Level Zero Ownership -- ------*-C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace level_zero {

// Since Level-Zero is not doing any reference counting itself, we have to
// be explicit about the ownership of the native handles used in the
// interop functions below.
//
enum class ownership { transfer, keep };

} // namespace level_zero
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
