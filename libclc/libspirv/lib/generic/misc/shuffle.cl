//===----------------- generic/lib/misc/shuffle.cl ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/misc/clc_shuffle.h>
#include <libspirv/spirv.h>

#define FUNCTION __spirv_ocl_shuffle
#define __CLC_FUNCTION(x) __clc_shuffle

#define __CLC_BODY <clc/misc/shuffle_def.inc>
#include <clc/integer/gentype.inc>

#define __CLC_BODY <clc/misc/shuffle_def.inc>
#include <clc/math/gentype.inc>
