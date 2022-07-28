//==------------- math.hpp - Intel specific math API -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The main header of Intel specific math API
//===----------------------------------------------------------------------===//

#pragma once

extern "C" {
float __imf_saturatef(float);
float __imf_copysignf(float, float);
double __imf_copysign(double, double);
};

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
float saturate(float x) { return __imf_saturatef(x); }

float copysign(float x, float y) { return __imf_copysignf(x, y); }
double copysign(double x, double y) { return __imf_copysign(x, y); }

} // namespace intel
} // namespace ext

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
