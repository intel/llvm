//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/integer/clc_mad_sat.h>
#include <libspirv/spirv.h>

#define __CLC_BODY <mad_sat.inc>
#include <clc/integer/gentype.inc>
