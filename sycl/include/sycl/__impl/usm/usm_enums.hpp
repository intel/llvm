//==-------------- usm_enums.hpp - SYCL USM Enums --------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
#else
namespace __sycl_internal {
inline namespace __v1 {
#endif
namespace usm {

enum class alloc { host = 0, device = 1, shared = 2, unknown = 3 };

} // namespace usm
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#ifndef __SYCL_ENABLE_SYCL121_NAMESPACE
namespace sycl {
  using namespace __sycl_internal::__v1;
}
#endif
