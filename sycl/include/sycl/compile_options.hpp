//==---- compile_options.hpp - SYCL compile options Enums -----*- C++ -*---==//
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

#define __SYCL_COMPILE_OPTION(OPTION, ID) OPTION = ID,
enum class __SYCL_TYPE(compile_options) compile_options {
#include <sycl/info/compile_options.def>
};
#undef __SYCL_COMPILE_OPTION

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
