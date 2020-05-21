//==-------------- backend_types.hpp - SYCL backend types ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

enum class backend : char { host, opencl, level0, cuda };

template <backend name, typename SYCLObjectT> struct interop;

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
