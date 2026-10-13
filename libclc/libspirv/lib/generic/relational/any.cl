//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/relational/clc_any.h"
#include "libspirv/spirv.h"

_CLC_OVERLOAD _CLC_DEF bool __spirv_Any(char2 x) { return __clc_any(x); }

_CLC_OVERLOAD _CLC_DEF bool __spirv_Any(char3 x) { return __clc_any(x); }

_CLC_OVERLOAD _CLC_DEF bool __spirv_Any(char4 x) { return __clc_any(x); }

_CLC_OVERLOAD _CLC_DEF bool __spirv_Any(char8 x) { return __clc_any(x); }

_CLC_OVERLOAD _CLC_DEF bool __spirv_Any(char16 x) { return __clc_any(x); }
