//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#define __CLC_BODY <fast_distance.inc>
#define __FLOAT_ONLY
#include <libspirv/generic/math/floatn.inc>
#undef __FLOAT_ONLY
