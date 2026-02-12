//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/utils.h>

#ifndef __CLC_BUILTIN
#define __CLC_BUILTIN __CLC_XCONCAT(__clc_, __CLC_FUNCTION)
#endif

#ifndef __CLC_BUILTIN_D
#define __CLC_BUILTIN_D __CLC_BUILTIN
#endif

#ifndef __CLC_BUILTIN_F
#define __CLC_BUILTIN_F __CLC_BUILTIN
#endif

#ifndef __CLC_BUILTIN_H
#define __CLC_BUILTIN_H __CLC_BUILTIN_F
#endif

#define __CLC_SCALAR
#define __CLC_MIN_VECSIZE 1

#define __CLC_IMPL_FUNCTION __CLC_BUILTIN_F
#define __CLC_GENTYPE float
#include <clc/shared/binary_def_scalarize.inc>
#undef __CLC_GENTYPE
#undef __CLC_IMPL_FUNCTION

#ifndef __CLC_FLOAT_ONLY

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __CLC_IMPL_FUNCTION __CLC_BUILTIN_D
#define __CLC_GENTYPE double
#include <clc/shared/binary_def_scalarize.inc>
#undef __CLC_GENTYPE
#undef __CLC_IMPL_FUNCTION

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define __CLC_IMPL_FUNCTION __CLC_BUILTIN_H
#define __CLC_GENTYPE half
#include <clc/shared/binary_def_scalarize.inc>
#undef __CLC_GENTYPE
#undef __CLC_IMPL_FUNCTION

#endif

#endif

#undef __CLC_SCALAR
#undef FUNCTION
#undef __CLC_MIN_VECSIZE
