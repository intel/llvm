//===-- lib/quadmath/jn.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "math-entries.h"

namespace Fortran::runtime {
extern "C" {

#if HAS_LDBL128 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDEF(JnF128)(
    int n, CppTypeFor<TypeCategory::Real, 16> x) {
  return Jn<true>::invoke(n, x);
}
#endif

} // extern "C"
} // namespace Fortran::runtime
