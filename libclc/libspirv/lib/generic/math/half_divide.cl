//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#define divide(x, y) (x / y)

#define __CLC_BUILTIN divide
#define __CLC_FUNCTION __spirv_ocl_half_divide
#include <math/binary_builtin.inc>
