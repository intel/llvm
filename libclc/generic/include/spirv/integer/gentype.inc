//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//These 2 defines only change when switching between data sizes or base types to
//keep this file manageable.
#define __SPIRV_GENSIZE 8
#define __SPIRV_SCALAR_GENTYPE char

#define __SPIRV_GENTYPE char
#define __SPIRV_U_GENTYPE uchar
#define __SPIRV_S_GENTYPE char
#define __SPIRV_SCALAR 1
#define __SPIRV_VECSIZE
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_SCALAR
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE char2
#define __SPIRV_U_GENTYPE uchar2
#define __SPIRV_S_GENTYPE char2
#define __SPIRV_VECSIZE 2
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE char3
#define __SPIRV_U_GENTYPE uchar3
#define __SPIRV_S_GENTYPE char3
#define __SPIRV_VECSIZE 3
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE char4
#define __SPIRV_U_GENTYPE uchar4
#define __SPIRV_S_GENTYPE char4
#define __SPIRV_VECSIZE 4
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE char8
#define __SPIRV_U_GENTYPE uchar8
#define __SPIRV_S_GENTYPE char8
#define __SPIRV_VECSIZE 8
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE char16
#define __SPIRV_U_GENTYPE uchar16
#define __SPIRV_S_GENTYPE char16
#define __SPIRV_VECSIZE 16
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#undef __SPIRV_SCALAR_GENTYPE
#define __SPIRV_SCALAR_GENTYPE uchar

#define __SPIRV_GENTYPE uchar
#define __SPIRV_U_GENTYPE uchar
#define __SPIRV_S_GENTYPE char
#define __SPIRV_SCALAR 1
#define __SPIRV_VECSIZE
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_SCALAR
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE uchar2
#define __SPIRV_U_GENTYPE uchar2
#define __SPIRV_S_GENTYPE char2
#define __SPIRV_VECSIZE 2
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE uchar3
#define __SPIRV_U_GENTYPE uchar3
#define __SPIRV_S_GENTYPE char3
#define __SPIRV_VECSIZE 3
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE uchar4
#define __SPIRV_U_GENTYPE uchar4
#define __SPIRV_S_GENTYPE char4
#define __SPIRV_VECSIZE 4
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE uchar8
#define __SPIRV_U_GENTYPE uchar8
#define __SPIRV_S_GENTYPE char8
#define __SPIRV_VECSIZE 8
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE uchar16
#define __SPIRV_U_GENTYPE uchar16
#define __SPIRV_S_GENTYPE char16
#define __SPIRV_VECSIZE 16
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#undef __SPIRV_GENSIZE
#define __SPIRV_GENSIZE 16
#undef __SPIRV_SCALAR_GENTYPE
#define __SPIRV_SCALAR_GENTYPE short

#define __SPIRV_GENTYPE short
#define __SPIRV_U_GENTYPE ushort
#define __SPIRV_S_GENTYPE short
#define __SPIRV_SCALAR 1
#define __SPIRV_VECSIZE
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_SCALAR
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE short2
#define __SPIRV_U_GENTYPE ushort2
#define __SPIRV_S_GENTYPE short2
#define __SPIRV_VECSIZE 2
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE short3
#define __SPIRV_U_GENTYPE ushort3
#define __SPIRV_S_GENTYPE short3
#define __SPIRV_VECSIZE 3
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE short4
#define __SPIRV_U_GENTYPE ushort4
#define __SPIRV_S_GENTYPE short4
#define __SPIRV_VECSIZE 4
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE short8
#define __SPIRV_U_GENTYPE ushort8
#define __SPIRV_S_GENTYPE short8
#define __SPIRV_VECSIZE 8
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE short16
#define __SPIRV_U_GENTYPE ushort16
#define __SPIRV_S_GENTYPE short16
#define __SPIRV_VECSIZE 16
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#undef __SPIRV_SCALAR_GENTYPE
#define __SPIRV_SCALAR_GENTYPE ushort

#define __SPIRV_GENTYPE ushort
#define __SPIRV_U_GENTYPE ushort
#define __SPIRV_S_GENTYPE short
#define __SPIRV_SCALAR 1
#define __SPIRV_VECSIZE
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_SCALAR
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE ushort2
#define __SPIRV_U_GENTYPE ushort2
#define __SPIRV_S_GENTYPE short2
#define __SPIRV_VECSIZE 2
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE ushort3
#define __SPIRV_U_GENTYPE ushort3
#define __SPIRV_S_GENTYPE short3
#define __SPIRV_VECSIZE 3
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE ushort4
#define __SPIRV_U_GENTYPE ushort4
#define __SPIRV_S_GENTYPE short4
#define __SPIRV_VECSIZE 4
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE ushort8
#define __SPIRV_U_GENTYPE ushort8
#define __SPIRV_S_GENTYPE short8
#define __SPIRV_VECSIZE 8
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE ushort16
#define __SPIRV_U_GENTYPE ushort16
#define __SPIRV_S_GENTYPE short16
#define __SPIRV_VECSIZE 16
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#undef __SPIRV_GENSIZE
#define __SPIRV_GENSIZE 32
#undef __SPIRV_SCALAR_GENTYPE
#define __SPIRV_SCALAR_GENTYPE int

#define __SPIRV_GENTYPE int
#define __SPIRV_U_GENTYPE uint
#define __SPIRV_S_GENTYPE int
#define __SPIRV_SCALAR 1
#define __SPIRV_VECSIZE
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_SCALAR
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE int2
#define __SPIRV_U_GENTYPE uint2
#define __SPIRV_S_GENTYPE int2
#define __SPIRV_VECSIZE 2
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE int3
#define __SPIRV_U_GENTYPE uint3
#define __SPIRV_S_GENTYPE int3
#define __SPIRV_VECSIZE 3
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE int4
#define __SPIRV_U_GENTYPE uint4
#define __SPIRV_S_GENTYPE int4
#define __SPIRV_VECSIZE 4
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE int8
#define __SPIRV_U_GENTYPE uint8
#define __SPIRV_S_GENTYPE int8
#define __SPIRV_VECSIZE 8
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE int16
#define __SPIRV_U_GENTYPE uint16
#define __SPIRV_S_GENTYPE int16
#define __SPIRV_VECSIZE 16
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#undef __SPIRV_SCALAR_GENTYPE
#define __SPIRV_SCALAR_GENTYPE uint

#define __SPIRV_GENTYPE uint
#define __SPIRV_U_GENTYPE uint
#define __SPIRV_S_GENTYPE int
#define __SPIRV_SCALAR 1
#define __SPIRV_VECSIZE
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_SCALAR
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE uint2
#define __SPIRV_U_GENTYPE uint2
#define __SPIRV_S_GENTYPE int2
#define __SPIRV_VECSIZE 2
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE uint3
#define __SPIRV_U_GENTYPE uint3
#define __SPIRV_S_GENTYPE int3
#define __SPIRV_VECSIZE 3
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE uint4
#define __SPIRV_U_GENTYPE uint4
#define __SPIRV_S_GENTYPE int4
#define __SPIRV_VECSIZE 4
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE uint8
#define __SPIRV_U_GENTYPE uint8
#define __SPIRV_S_GENTYPE int8
#define __SPIRV_VECSIZE 8
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE uint16
#define __SPIRV_U_GENTYPE uint16
#define __SPIRV_S_GENTYPE int16
#define __SPIRV_VECSIZE 16
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#undef __SPIRV_GENSIZE
#define __SPIRV_GENSIZE 64
#undef __SPIRV_SCALAR_GENTYPE
#define __SPIRV_SCALAR_GENTYPE long

#define __SPIRV_GENTYPE long
#define __SPIRV_U_GENTYPE ulong
#define __SPIRV_S_GENTYPE long
#define __SPIRV_SCALAR 1
#define __SPIRV_VECSIZE
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_SCALAR
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE long2
#define __SPIRV_U_GENTYPE ulong2
#define __SPIRV_S_GENTYPE long2
#define __SPIRV_VECSIZE 2
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE long3
#define __SPIRV_U_GENTYPE ulong3
#define __SPIRV_S_GENTYPE long3
#define __SPIRV_VECSIZE 3
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE long4
#define __SPIRV_U_GENTYPE ulong4
#define __SPIRV_S_GENTYPE long4
#define __SPIRV_VECSIZE 4
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE long8
#define __SPIRV_U_GENTYPE ulong8
#define __SPIRV_S_GENTYPE long8
#define __SPIRV_VECSIZE 8
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE long16
#define __SPIRV_U_GENTYPE ulong16
#define __SPIRV_S_GENTYPE long16
#define __SPIRV_VECSIZE 16
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#undef __SPIRV_SCALAR_GENTYPE
#define __SPIRV_SCALAR_GENTYPE ulong

#define __SPIRV_GENTYPE ulong
#define __SPIRV_U_GENTYPE ulong
#define __SPIRV_S_GENTYPE long
#define __SPIRV_SCALAR 1
#define __SPIRV_VECSIZE
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_SCALAR
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE ulong2
#define __SPIRV_U_GENTYPE ulong2
#define __SPIRV_S_GENTYPE long2
#define __SPIRV_VECSIZE 2
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE ulong3
#define __SPIRV_U_GENTYPE ulong3
#define __SPIRV_S_GENTYPE long3
#define __SPIRV_VECSIZE 3
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE ulong4
#define __SPIRV_U_GENTYPE ulong4
#define __SPIRV_S_GENTYPE long4
#define __SPIRV_VECSIZE 4
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE ulong8
#define __SPIRV_U_GENTYPE ulong8
#define __SPIRV_S_GENTYPE long8
#define __SPIRV_VECSIZE 8
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#define __SPIRV_GENTYPE ulong16
#define __SPIRV_U_GENTYPE ulong16
#define __SPIRV_S_GENTYPE long16
#define __SPIRV_VECSIZE 16
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_U_GENTYPE
#undef __SPIRV_S_GENTYPE

#undef __SPIRV_GENSIZE
#undef __SPIRV_SCALAR_GENTYPE
#undef __SPIRV_BODY
