//==-------- spirv_ops_base.hpp --- SPIRV operation common support --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_types.hpp>       // for Scope, __ocl_event_t
#include <sycl/detail/defines_elementary.hpp> // for __DPCPP_SYCL_EXTERNAL
#include <sycl/detail/export.hpp>             // for __SYCL_EXPORT

// Convergent attribute
#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_CONVERGENT__ __attribute__((convergent))
#else
#define __SYCL_CONVERGENT__
#endif