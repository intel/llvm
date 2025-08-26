//==--- devicelib_combine.cpp - merge multiple sycl devicelibs into one ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "cmath_wrapper.cpp"
#include "fallback-cmath.cpp"
#if defined(_WIN32)
#include "msvc_math.hpp"
#endif
#include "cmath_wrapper_fp64.cpp"
#include "fallback-cmath-fp64.cpp"
#include "complex_wrapper.cpp"
#include "fallback-complex.cpp"
#include "complex_wrapper_fp64.cpp"
#include "fallback-complex-fp64.cpp"
#include "crt_wrapper.cpp"
#include "fallback-cassert.cpp"
#include "fallback-cstring.cpp"
#include "imf_wrapper.cpp"
#include "imf_wrapper_fp64.cpp"
#include "imf_wrapper_bf16.cpp"
