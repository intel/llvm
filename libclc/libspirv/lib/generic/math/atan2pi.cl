//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>
#include <clc/math/clc_atan2pi.h>

#define FUNCTION __spirv_ocl_atan2pi
#define __IMPL_FUNCTION(x) __clc_atan2pi
#define __CLC_BODY <clc/shared/binary_def.inc>

#include <clc/math/gentype.inc>
