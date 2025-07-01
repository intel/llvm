//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <clc/math/tables.h>

#define FUNCTION __spirv_ocl_log10

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif // cl_khr_fp64

#define COMPILING_LOG10
#include "log_base.h"
#undef COMPILING_LOG10

#define __FLOAT_ONLY
#define __IMPL_FUNCTION __spirv_ocl_log10
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __IMPL_FUNCTION
#undef __FLOAT_ONLY

#ifdef cl_khr_fp64

#define __DOUBLE_ONLY
#define __IMPL_FUNCTION __spirv_ocl_log10
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __IMPL_FUNCTION
#undef __DOUBLE_ONLY

#endif // cl_khr_fp64

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define __CLC_MIN_VECSIZE 1
#define __HALF_ONLY
#define __IMPL_FUNCTION __builtin_log10f16
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __IMPL_FUNCTION
#undef __HALF_ONLY
#undef __CLC_MIN_VECSIZE

#endif

#undef FUNCTION
