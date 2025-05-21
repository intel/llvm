//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_subnormal_config.h>
#include <clc/math/clc_pow.h>
#include <clc/math/math.h>

#define __CLC_BODY <pow.inc>
#include <clc/math/gentype.inc>
