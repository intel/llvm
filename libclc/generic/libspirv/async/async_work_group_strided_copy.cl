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

#define __CLC_GENTYPE int2
#define __CLC_GENTYPE_MANGLED Dv2_i
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int4
#define __CLC_GENTYPE_MANGLED Dv4_i
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint2
#define __CLC_GENTYPE_MANGLED Dv2_j
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint4
#define __CLC_GENTYPE_MANGLED Dv4_j
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE char4
#define __CLC_GENTYPE_MANGLED Dv4_c
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE char8
#define __CLC_GENTYPE_MANGLED Dv8_c
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE char16
#define __CLC_GENTYPE_MANGLED Dv16_c
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#ifndef __CLC_NO_SCHAR

#define __CLC_GENTYPE schar4
#define __CLC_GENTYPE_MANGLED Dv4_a
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE schar8
#define __CLC_GENTYPE_MANGLED Dv8_a
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE schar16
#define __CLC_GENTYPE_MANGLED Dv16_a
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#endif //__CLC_NO_SCHAR

#define __CLC_GENTYPE uchar4
#define __CLC_GENTYPE_MANGLED Dv4_h
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uchar8
#define __CLC_GENTYPE_MANGLED Dv8_h
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uchar16
#define __CLC_GENTYPE_MANGLED Dv16_h
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE short2
#define __CLC_GENTYPE_MANGLED Dv2_s
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE short4
#define __CLC_GENTYPE_MANGLED Dv4_s
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE short8
#define __CLC_GENTYPE_MANGLED Dv8_s
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE double
#define __CLC_GENTYPE_MANGLED d
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE double2
#define __CLC_GENTYPE_MANGLED Dv2_d
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float
#define __CLC_GENTYPE_MANGLED f
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float2
#define __CLC_GENTYPE_MANGLED Dv2_f
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float4
#define __CLC_GENTYPE_MANGLED Dv4_f
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE long
#define __CLC_GENTYPE_MANGLED l
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE long2
#define __CLC_GENTYPE_MANGLED Dv2_l
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

#define __CLC_GENTYPE ulong2
#define __CLC_GENTYPE_MANGLED Dv2_m
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ushort2
#define __CLC_GENTYPE_MANGLED Dv2_t
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ushort4
#define __CLC_GENTYPE_MANGLED Dv4_t
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ushort8
#define __CLC_GENTYPE_MANGLED Dv8_t
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16: enable

#define __CLC_GENTYPE half2
#define __CLC_GENTYPE_MANGLED Dv2_Dh
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE half4
#define __CLC_GENTYPE_MANGLED Dv4_Dh
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE half8
#define __CLC_GENTYPE_MANGLED Dv8_Dh
#include <async_work_group_strided_copy.inc>
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#endif
