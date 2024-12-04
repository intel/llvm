//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#include <clc/clcmacro.h>

#define __CLC_BODY <bitselect.inc>
#include <clc/integer/gentype.inc>
#undef __CLC_BODY

#define FLOAT_BITSELECT(f_type, i_type, width)                                 \
  _CLC_OVERLOAD _CLC_DEF f_type##width __spirv_ocl_bitselect(                  \
      f_type##width x, f_type##width y, f_type##width z) {                     \
    return as_##f_type##width(__spirv_ocl_bitselect(                           \
        as_##i_type##width(x), as_##i_type##width(y), as_##i_type##width(z))); \
  }

FLOAT_BITSELECT(float, uint, )
FLOAT_BITSELECT(float, uint, 2)
FLOAT_BITSELECT(float, uint, 3)
FLOAT_BITSELECT(float, uint, 4)
FLOAT_BITSELECT(float, uint, 8)
FLOAT_BITSELECT(float, uint, 16)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

FLOAT_BITSELECT(double, ulong, )
FLOAT_BITSELECT(double, ulong, 2)
FLOAT_BITSELECT(double, ulong, 3)
FLOAT_BITSELECT(double, ulong, 4)
FLOAT_BITSELECT(double, ulong, 8)
FLOAT_BITSELECT(double, ulong, 16)

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

FLOAT_BITSELECT(half, ushort, )
FLOAT_BITSELECT(half, ushort, 2)
FLOAT_BITSELECT(half, ushort, 3)
FLOAT_BITSELECT(half, ushort, 4)
FLOAT_BITSELECT(half, ushort, 8)
FLOAT_BITSELECT(half, ushort, 16)

#endif
