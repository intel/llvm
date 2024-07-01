//==------------ misc_intrin.hpp - SYCL Kernel Properties -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Declares miscellaneous SYCL intrinsics.
//===----------------------------------------------------------------------===//

#pragma once

/// @cond SYCL_DETAIL

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/detail/defines_elementary.hpp>
#define __SYCL_INTRIN __DPCPP_SYCL_EXTERNAL
#else
#define __SYCL_INTRIN inline
#endif // __SYCL_DEVICE_ONLY__

__SYCL_INTRIN void __sycl_set_kernel_properties(int prop_mask)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
} // Only "double GRF" property is supported for now, safe to ignore on host.
#endif // __SYCL_DEVICE_ONLY__

/// @endcond SYCL_DETAIL
