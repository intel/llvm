//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcfunc.h>
#include <clc/clctypes.h>
#include <clc/utils.h>

#define __CLC_AS_GENTYPE __CLC_XCONCAT(__clc_as_, __CLC_GENTYPE)
#define __CLC_CONVERT_GENTYPE __CLC_XCONCAT(__clc_convert_, __CLC_GENTYPE)

#define __CLC_AS_U_GENTYPE __CLC_XCONCAT(__clc_as_, __CLC_U_GENTYPE)
#define __CLC_CONVERT_U_GENTYPE __CLC_XCONCAT(__clc_convert_, __CLC_U_GENTYPE)

#define __CLC_AS_S_GENTYPE __CLC_XCONCAT(__clc_as_, __CLC_S_GENTYPE)
#define __CLC_CONVERT_S_GENTYPE __CLC_XCONCAT(__clc_convert_, __CLC_S_GENTYPE)

// These 2 defines only change when switching between data sizes or base types
// to keep this file manageable.
#define __CLC_GENSIZE 8
#define __CLC_SCALAR_GENTYPE char
#define __CLC_GEN_S

#define __CLC_GENTYPE char
#define __CLC_U_GENTYPE uchar
#define __CLC_S_GENTYPE char
#define __CLC_SCALAR
#define __CLC_VECSIZE
#define __CLC_VECSIZE_OR_1 1
#include __CLC_BODY
#undef __CLC_VECSIZE_OR_1
#undef __CLC_VECSIZE
#undef __CLC_SCALAR
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_VECSIZE_OR_1 __CLC_VECSIZE

#define __CLC_GENTYPE char2
#define __CLC_U_GENTYPE uchar2
#define __CLC_S_GENTYPE char2
#define __CLC_VECSIZE 2
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE char3
#define __CLC_U_GENTYPE uchar3
#define __CLC_S_GENTYPE char3
#define __CLC_VECSIZE 3
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE char4
#define __CLC_U_GENTYPE uchar4
#define __CLC_S_GENTYPE char4
#define __CLC_VECSIZE 4
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE char8
#define __CLC_U_GENTYPE uchar8
#define __CLC_S_GENTYPE char8
#define __CLC_VECSIZE 8
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE char16
#define __CLC_U_GENTYPE uchar16
#define __CLC_S_GENTYPE char16
#define __CLC_VECSIZE 16
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE
#undef __CLC_VECSIZE_OR_1

#undef __CLC_SCALAR_GENTYPE
#define __CLC_SCALAR_GENTYPE uchar
#undef __CLC_GEN_S
#define __CLC_GEN_U

#define __CLC_GENTYPE uchar
#define __CLC_U_GENTYPE uchar
#define __CLC_S_GENTYPE char
#define __CLC_SCALAR
#define __CLC_VECSIZE
#define __CLC_VECSIZE_OR_1 1
#include __CLC_BODY
#undef __CLC_VECSIZE_OR_1
#undef __CLC_VECSIZE
#undef __CLC_SCALAR
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_VECSIZE_OR_1 __CLC_VECSIZE

#define __CLC_GENTYPE uchar2
#define __CLC_U_GENTYPE uchar2
#define __CLC_S_GENTYPE char2
#define __CLC_VECSIZE 2
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE uchar3
#define __CLC_U_GENTYPE uchar3
#define __CLC_S_GENTYPE char3
#define __CLC_VECSIZE 3
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE uchar4
#define __CLC_U_GENTYPE uchar4
#define __CLC_S_GENTYPE char4
#define __CLC_VECSIZE 4
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE uchar8
#define __CLC_U_GENTYPE uchar8
#define __CLC_S_GENTYPE char8
#define __CLC_VECSIZE 8
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE uchar16
#define __CLC_U_GENTYPE uchar16
#define __CLC_S_GENTYPE char16
#define __CLC_VECSIZE 16
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE
#undef __CLC_VECSIZE_OR_1

#undef __CLC_GENSIZE
#define __CLC_GENSIZE 16
#undef __CLC_SCALAR_GENTYPE
#define __CLC_SCALAR_GENTYPE short
#undef __CLC_GEN_U
#define __CLC_GEN_S

#define __CLC_GENTYPE short
#define __CLC_U_GENTYPE ushort
#define __CLC_S_GENTYPE short
#define __CLC_SCALAR
#define __CLC_VECSIZE
#define __CLC_VECSIZE_OR_1 1
#include __CLC_BODY
#undef __CLC_VECSIZE_OR_1
#undef __CLC_VECSIZE
#undef __CLC_SCALAR
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_VECSIZE_OR_1 __CLC_VECSIZE

#define __CLC_GENTYPE short2
#define __CLC_U_GENTYPE ushort2
#define __CLC_S_GENTYPE short2
#define __CLC_VECSIZE 2
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE short3
#define __CLC_U_GENTYPE ushort3
#define __CLC_S_GENTYPE short3
#define __CLC_VECSIZE 3
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE short4
#define __CLC_U_GENTYPE ushort4
#define __CLC_S_GENTYPE short4
#define __CLC_VECSIZE 4
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE short8
#define __CLC_U_GENTYPE ushort8
#define __CLC_S_GENTYPE short8
#define __CLC_VECSIZE 8
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE short16
#define __CLC_U_GENTYPE ushort16
#define __CLC_S_GENTYPE short16
#define __CLC_VECSIZE 16
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE
#undef __CLC_VECSIZE_OR_1

#undef __CLC_SCALAR_GENTYPE
#define __CLC_SCALAR_GENTYPE ushort
#undef __CLC_GEN_S
#define __CLC_GEN_U

#define __CLC_GENTYPE ushort
#define __CLC_U_GENTYPE ushort
#define __CLC_S_GENTYPE short
#define __CLC_SCALAR
#define __CLC_VECSIZE
#define __CLC_VECSIZE_OR_1 1
#include __CLC_BODY
#undef __CLC_VECSIZE_OR_1
#undef __CLC_VECSIZE
#undef __CLC_SCALAR
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_VECSIZE_OR_1 __CLC_VECSIZE

#define __CLC_GENTYPE ushort2
#define __CLC_U_GENTYPE ushort2
#define __CLC_S_GENTYPE short2
#define __CLC_VECSIZE 2
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE ushort3
#define __CLC_U_GENTYPE ushort3
#define __CLC_S_GENTYPE short3
#define __CLC_VECSIZE 3
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE ushort4
#define __CLC_U_GENTYPE ushort4
#define __CLC_S_GENTYPE short4
#define __CLC_VECSIZE 4
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE ushort8
#define __CLC_U_GENTYPE ushort8
#define __CLC_S_GENTYPE short8
#define __CLC_VECSIZE 8
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE ushort16
#define __CLC_U_GENTYPE ushort16
#define __CLC_S_GENTYPE short16
#define __CLC_VECSIZE 16
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE
#undef __CLC_VECSIZE_OR_1

#undef __CLC_GENSIZE
#define __CLC_GENSIZE 32
#undef __CLC_SCALAR_GENTYPE
#define __CLC_SCALAR_GENTYPE int
#undef __CLC_GEN_U
#define __CLC_GEN_S

#define __CLC_GENTYPE int
#define __CLC_U_GENTYPE uint
#define __CLC_S_GENTYPE int
#define __CLC_SCALAR
#define __CLC_VECSIZE
#define __CLC_VECSIZE_OR_1 1
#include __CLC_BODY
#undef __CLC_VECSIZE_OR_1
#undef __CLC_VECSIZE
#undef __CLC_SCALAR
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_VECSIZE_OR_1 __CLC_VECSIZE

#define __CLC_GENTYPE int2
#define __CLC_U_GENTYPE uint2
#define __CLC_S_GENTYPE int2
#define __CLC_VECSIZE 2
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE int3
#define __CLC_U_GENTYPE uint3
#define __CLC_S_GENTYPE int3
#define __CLC_VECSIZE 3
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE int4
#define __CLC_U_GENTYPE uint4
#define __CLC_S_GENTYPE int4
#define __CLC_VECSIZE 4
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE int8
#define __CLC_U_GENTYPE uint8
#define __CLC_S_GENTYPE int8
#define __CLC_VECSIZE 8
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE int16
#define __CLC_U_GENTYPE uint16
#define __CLC_S_GENTYPE int16
#define __CLC_VECSIZE 16
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE
#undef __CLC_VECSIZE_OR_1

#undef __CLC_SCALAR_GENTYPE
#define __CLC_SCALAR_GENTYPE uint
#undef __CLC_GEN_S
#define __CLC_GEN_U

#define __CLC_GENTYPE uint
#define __CLC_U_GENTYPE uint
#define __CLC_S_GENTYPE int
#define __CLC_SCALAR
#define __CLC_VECSIZE
#define __CLC_VECSIZE_OR_1 1
#include __CLC_BODY
#undef __CLC_VECSIZE_OR_1
#undef __CLC_VECSIZE
#undef __CLC_SCALAR
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_VECSIZE_OR_1 __CLC_VECSIZE

#define __CLC_GENTYPE uint2
#define __CLC_U_GENTYPE uint2
#define __CLC_S_GENTYPE int2
#define __CLC_VECSIZE 2
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE uint3
#define __CLC_U_GENTYPE uint3
#define __CLC_S_GENTYPE int3
#define __CLC_VECSIZE 3
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE uint4
#define __CLC_U_GENTYPE uint4
#define __CLC_S_GENTYPE int4
#define __CLC_VECSIZE 4
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE uint8
#define __CLC_U_GENTYPE uint8
#define __CLC_S_GENTYPE int8
#define __CLC_VECSIZE 8
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE uint16
#define __CLC_U_GENTYPE uint16
#define __CLC_S_GENTYPE int16
#define __CLC_VECSIZE 16
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE
#undef __CLC_VECSIZE_OR_1

#undef __CLC_GENSIZE
#define __CLC_GENSIZE 64
#undef __CLC_SCALAR_GENTYPE
#define __CLC_SCALAR_GENTYPE long
#undef __CLC_GEN_U
#define __CLC_GEN_S

#define __CLC_GENTYPE long
#define __CLC_U_GENTYPE ulong
#define __CLC_S_GENTYPE long
#define __CLC_SCALAR
#define __CLC_VECSIZE
#define __CLC_VECSIZE_OR_1 1
#include __CLC_BODY
#undef __CLC_VECSIZE_OR_1
#undef __CLC_VECSIZE
#undef __CLC_SCALAR
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_VECSIZE_OR_1 __CLC_VECSIZE

#define __CLC_GENTYPE long2
#define __CLC_U_GENTYPE ulong2
#define __CLC_S_GENTYPE long2
#define __CLC_VECSIZE 2
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE long3
#define __CLC_U_GENTYPE ulong3
#define __CLC_S_GENTYPE long3
#define __CLC_VECSIZE 3
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE long4
#define __CLC_U_GENTYPE ulong4
#define __CLC_S_GENTYPE long4
#define __CLC_VECSIZE 4
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE long8
#define __CLC_U_GENTYPE ulong8
#define __CLC_S_GENTYPE long8
#define __CLC_VECSIZE 8
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE long16
#define __CLC_U_GENTYPE ulong16
#define __CLC_S_GENTYPE long16
#define __CLC_VECSIZE 16
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE
#undef __CLC_VECSIZE_OR_1

#undef __CLC_SCALAR_GENTYPE
#define __CLC_SCALAR_GENTYPE ulong
#undef __CLC_GEN_S
#define __CLC_GEN_U

#define __CLC_GENTYPE ulong
#define __CLC_U_GENTYPE ulong
#define __CLC_S_GENTYPE long
#define __CLC_SCALAR
#define __CLC_VECSIZE
#define __CLC_VECSIZE_OR_1 1
#include __CLC_BODY
#undef __CLC_VECSIZE_OR_1
#undef __CLC_VECSIZE_OR_1
#undef __CLC_VECSIZE
#undef __CLC_SCALAR
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_VECSIZE_OR_1 __CLC_VECSIZE

#define __CLC_GENTYPE ulong2
#define __CLC_U_GENTYPE ulong2
#define __CLC_S_GENTYPE long2
#define __CLC_VECSIZE 2
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE ulong3
#define __CLC_U_GENTYPE ulong3
#define __CLC_S_GENTYPE long3
#define __CLC_VECSIZE 3
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE ulong4
#define __CLC_U_GENTYPE ulong4
#define __CLC_S_GENTYPE long4
#define __CLC_VECSIZE 4
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE ulong8
#define __CLC_U_GENTYPE ulong8
#define __CLC_S_GENTYPE long8
#define __CLC_VECSIZE 8
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE ulong16
#define __CLC_U_GENTYPE ulong16
#define __CLC_S_GENTYPE long16
#define __CLC_VECSIZE 16
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE
#undef __CLC_VECSIZE_OR_1

#undef __CLC_GEN_U

#undef __CLC_GENSIZE
#undef __CLC_SCALAR_GENTYPE
#undef __CLC_BODY

#undef __CLC_CONVERT_S_GENTYPE
#undef __CLC_AS_S_GENTYPE

#undef __CLC_CONVERT_U_GENTYPE
#undef __CLC_AS_U_GENTYPE

#undef __CLC_CONVERT_GENTYPE
#undef __CLC_AS_GENTYPE
