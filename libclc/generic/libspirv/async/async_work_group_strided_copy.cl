//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/async/common.h>
#include <spirv/spirv.h>

#define __CLC_BODY <async_work_group_strided_copy.inc>
#define __CLC_GEN_VEC3
#include <clc/async/gentype.inc>

#define __CLC_GENTYPE double
#define __CLC_GENTYPE_MANGLED d
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float
#define __CLC_GENTYPE_MANGLED f
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE long
#define __CLC_GENTYPE_MANGLED l
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int
#define __CLC_GENTYPE_MANGLED i
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint
#define __CLC_GENTYPE_MANGLED j
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ulong
#define __CLC_GENTYPE_MANGLED m
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE
