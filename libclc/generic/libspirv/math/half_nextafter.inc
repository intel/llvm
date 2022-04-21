//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HALF_NEXTAFTER_INC
#define HALF_NEXTAFTER_INC

#include <clcmacro.h>
#include <math/math.h>
#include <spirv/spirv.h>

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __spirv_ocl_nextafter(half x, half y) {
  // NaNs
  if (x != x)
    return x;
  if (y != y)
    return y;
  // Parity
  if (x == y)
    return x;

  short *a = (short *)&x;
  short *b = (short *)&y;
  // Checking for sign digit
  if (*a & 0x8000)
    *a = 0x8000 - *a;
  if (*b & 0x8000)
    *b = 0x8000 - *b;
  // Increment / decrement
  *a += (*a < *b) ? 1 : -1;
  // Undo the sign flip if necessary
  *a = (*a < 0) ? 0x8000 - *a : *a;
  return x;
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_nextafter, half,
                      half)

#endif

#endif
