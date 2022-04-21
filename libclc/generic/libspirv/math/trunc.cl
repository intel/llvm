//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clcmacro.h>
#include <spirv/spirv.h>

// Map the llvm intrinsic to an OpenCL function.
#define __CLC_FUNCTION __clc___spirv_ocl_trunc
#define __CLC_INTRINSIC "llvm.trunc"
#include <math/unary_intrin.inc>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION __spirv_ocl_trunc
#include <math/unary_builtin.inc>
