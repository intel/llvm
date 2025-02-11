//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>
#include <integer/popcount.h>

#define __CLC_FUNC __spirv_ocl_popcount
#define __CLC_IMPL_FUNC __clc_native_popcount

#define __CLC_BODY "../../../../generic/lib/clc_unary.inc"
#include <clc/integer/gentype.inc>
