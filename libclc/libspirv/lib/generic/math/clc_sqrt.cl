//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

// Map the llvm sqrt intrinsic to an OpenCL function.
#define __CLC_FUNCTION __clc_llvm_intr_sqrt
#define __CLC_INTRINSIC "llvm.sqrt"
#include <math/unary_intrin.inc>
#undef __CLC_FUNCTION
#undef __CLC_INTRINSIC

#define __CLC_BODY <clc_sqrt_impl.inc>
#include <clc/math/gentype.inc>
