#define __CLC_GEN_S
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_GENTYPE int
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int2
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int3
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int4
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int8
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int16
#include __CLC_BODY
#undef __CLC_GENTYPE

#undef __CLC_GEN_S

#define __CLC_GENTYPE uint
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint2
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint3
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint4
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint8
#include __CLC_BODY
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint16
#include __CLC_BODY
#undef __CLC_GENTYPE
