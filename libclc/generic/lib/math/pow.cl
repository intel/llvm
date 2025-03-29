//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
<<<<<<< HEAD
#include <libspirv/spirv.h>

#define __CLC_BODY <pow.inc>
=======
#include <clc/math/clc_pow.h>

#define FUNCTION pow
#define __CLC_BODY <clc/shared/binary_def.inc>
>>>>>>> b52977b868b02625ade1f14bfbe835e299b26f0e
#include <clc/math/gentype.inc>
