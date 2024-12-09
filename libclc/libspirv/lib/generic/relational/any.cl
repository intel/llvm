//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#define _CLC_ANY(v) (((v) >> ((sizeof(v) * 8) - 1)) & 0x1)
#define _CLC_ANY2(v) (_CLC_ANY((v).s0) | _CLC_ANY((v).s1))
#define _CLC_ANY3(v) (_CLC_ANY2((v)) | _CLC_ANY((v).s2))
#define _CLC_ANY4(v) (_CLC_ANY3((v)) | _CLC_ANY((v).s3))
#define _CLC_ANY8(v)                                                           \
  (_CLC_ANY4((v)) | _CLC_ANY((v).s4) | _CLC_ANY((v).s5) | _CLC_ANY((v).s6) |   \
   _CLC_ANY((v).s7))
#define _CLC_ANY16(v)                                                          \
  (_CLC_ANY8((v)) | _CLC_ANY((v).s8) | _CLC_ANY((v).s9) | _CLC_ANY((v).sA) |   \
   _CLC_ANY((v).sB) | _CLC_ANY((v).sC) | _CLC_ANY((v).sD) | _CLC_ANY((v).sE) | \
   _CLC_ANY((v).sf))

#define ANY_ID(TYPE) _CLC_OVERLOAD _CLC_DEF bool __spirv_Any(TYPE v)

bool __spirv_Any(bool v) { return v; }

#define ANY_VECTORIZE(TYPE)                                                    \
  ANY_ID(TYPE##2) { return _CLC_ANY2(v); }                                     \
  ANY_ID(TYPE##3) { return _CLC_ANY3(v); }                                     \
  ANY_ID(TYPE##4) { return _CLC_ANY4(v); }                                     \
  ANY_ID(TYPE##8) { return _CLC_ANY8(v); }                                     \
  ANY_ID(TYPE##16) { return _CLC_ANY16(v); }

ANY_VECTORIZE(schar)
