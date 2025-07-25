//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/geometric/clc_fast_length.h>
#include <libspirv/spirv.h>

#define __FLOAT_ONLY
#define __CLC_BODY <fast_length.inc>
#include <clc/math/gentype.inc>
