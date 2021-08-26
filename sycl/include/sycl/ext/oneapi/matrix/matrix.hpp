//==------------------ matrix.hpp - SYCL matrix ----------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
/// Currently, this is the compilation command line needed to invoke AMX unit of
/// Sapphire Rapids CPU: clang++ -fsycl -march=sapphirerapids
/// fsycl-targets="spir64_x86_64-uknown-linux-sycldevice" -O2 main.cpp
///
///
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/feature_test.hpp>

#if (SYCL_EXT_ONEAPI_MATRIX == 1)
#if defined(__AMXTILE__) && defined(__AMXINT8__) && defined(__AMXBF16__)
#include <sycl/ext/oneapi/matrix/matrix-aot-amx.hpp>
#endif
#endif
#if (SYCL_EXT_ONEAPI_MATRIX == 2)
#include <sycl/ext/oneapi/matrix/matrix-jit.hpp>
#endif
