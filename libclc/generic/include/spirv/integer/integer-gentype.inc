//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __SPIRV_GENTYPE int
#include __SPIRV_BODY
#undef __SPIRV_GENTYPE

#define __SPIRV_GENTYPE int2
#include __SPIRV_BODY
#undef __SPIRV_GENTYPE

#define __SPIRV_GENTYPE int3
#include __SPIRV_BODY
#undef __SPIRV_GENTYPE

#define __SPIRV_GENTYPE int4
#include __SPIRV_BODY
#undef __SPIRV_GENTYPE

#define __SPIRV_GENTYPE int8
#include __SPIRV_BODY
#undef __SPIRV_GENTYPE

#define __SPIRV_GENTYPE int16
#include __SPIRV_BODY
#undef __SPIRV_GENTYPE

#define __SPIRV_GENTYPE uint
#include __SPIRV_BODY
#undef __SPIRV_GENTYPE

#define __SPIRV_GENTYPE uint2
#include __SPIRV_BODY
#undef __SPIRV_GENTYPE

#define __SPIRV_GENTYPE uint3
#include __SPIRV_BODY
#undef __SPIRV_GENTYPE

#define __SPIRV_GENTYPE uint4
#include __SPIRV_BODY
#undef __SPIRV_GENTYPE

#define __SPIRV_GENTYPE uint8
#include __SPIRV_BODY
#undef __SPIRV_GENTYPE

#define __SPIRV_GENTYPE uint16
#include __SPIRV_BODY
#undef __SPIRV_GENTYPE
