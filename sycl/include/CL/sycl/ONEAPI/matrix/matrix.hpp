//==------------------ matrix.hpp - SYCL matrix ----------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/sycl/detail/defines_elementary.hpp>
#include <CL/sycl/feature_test.hpp>

#if (SYCL_EXT_ONEAPI_MATRIX == 1)
#if defined(__AMXTILE__) && defined(__AMXINT8__) && defined(__AMXBF16__)
#include <CL/sycl/ONEAPI/matrix/matrix-aot-amx.hpp>
#endif
#endif
#if (SYCL_EXT_ONEAPI_MATRIX == 2)
#include <CL/sycl/ONEAPI/matrix/matrix-jit.hpp>
#endif
