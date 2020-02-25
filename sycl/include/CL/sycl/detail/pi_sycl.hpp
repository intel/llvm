//==---------------- pi_sycl.hpp - SYCL wrapper for PI ---------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/detail/defines.hpp>
#include <pi/pi.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

namespace RT = ::pi;

namespace detail {

namespace RT = ::pi;
using PiApiKind = ::PiApiKind;
namespace pi {
using namespace ::pi;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
