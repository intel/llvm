//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#define __CLC_BUILTIN __spirv_ocl_powr
#define __CLC_FUNCTION __spirv_ocl_half_powr
#include <math/binary_builtin.inc>
