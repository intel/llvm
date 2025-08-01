//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/math/clc_log.h>

float __nv_logf(float);
double __nv_log(double);

#define __CLC_FUNCTION __clc_log
#define __CLC_BUILTIN __nv_log
#define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, f)
#include <clc/math/unary_builtin_scalarize.inc>
