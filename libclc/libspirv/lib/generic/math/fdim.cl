//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <clc/math/math.h>

#define __CLC_BODY <fdim.inc>
#include <clc/math/gentype.inc>

#include <clc/clcmacro.h>

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_BINARY_BUILTIN(half, __spirv_ocl_fdim, __builtin_fdimf, half, half)

#endif
