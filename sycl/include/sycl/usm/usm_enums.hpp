//==-------------- usm_enums.hpp - SYCL USM Enums --------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace usm {

enum class alloc { host = 0, device = 1, shared = 2, unknown = 3 };

} // namespace usm
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
