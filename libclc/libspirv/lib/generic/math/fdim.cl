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

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define __HALF_ONLY
#define FUNCTION __spirv_ocl_fdim
#define __IMPL_FUNCTION __builtin_fdimf
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>

#endif
