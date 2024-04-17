//===----------------------- exception_helper.hp --------------------------===//
//
// This file is intended be used to break dependency cycles due to
// exception.hpp. In some header files where exceptions are thrown, it has to
// #include <sycl/exception.hpp> which could lead to a cyclic dependency. In
// that case, it can utilize this work-around. This helper class is not intended
// to replace all throwing exception cases.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <sycl/detail/pi.h>

namespace sycl {
inline namespace _V1 {
namespace detail {
__SYCL_EXPORT void throw_invalid_parameter(const char *Msg,
                                           const pi_int32 PIErr);
} // namespace detail
} // namespace _V1
} // namespace sycl
