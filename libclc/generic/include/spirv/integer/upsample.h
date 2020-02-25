//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __SPIRV_UPSAMPLE_DECL(BGENTYPE, GENTYPE, UGENTYPE) \
    _CLC_OVERLOAD _CLC_DECL BGENTYPE __spirv_ocl_u_upsample(GENTYPE hi, UGENTYPE lo);

#define __SPIRV_UPSAMPLE_VEC(BGENTYPE, GENTYPE, UGENTYPE) \
    __SPIRV_UPSAMPLE_DECL(BGENTYPE, GENTYPE, UGENTYPE) \
    __SPIRV_UPSAMPLE_DECL(BGENTYPE##2, GENTYPE##2, UGENTYPE##2) \
    __SPIRV_UPSAMPLE_DECL(BGENTYPE##3, GENTYPE##3, UGENTYPE##3) \
    __SPIRV_UPSAMPLE_DECL(BGENTYPE##4, GENTYPE##4, UGENTYPE##4) \
    __SPIRV_UPSAMPLE_DECL(BGENTYPE##8, GENTYPE##8, UGENTYPE##8) \
    __SPIRV_UPSAMPLE_DECL(BGENTYPE##16, GENTYPE##16, UGENTYPE##16) \

#define __SPIRV_UPSAMPLE_TYPES() \
    __SPIRV_UPSAMPLE_VEC(short, char, uchar) \
    __SPIRV_UPSAMPLE_VEC(ushort, uchar, uchar) \
    __SPIRV_UPSAMPLE_VEC(int, short, ushort) \
    __SPIRV_UPSAMPLE_VEC(uint, ushort, ushort) \
    __SPIRV_UPSAMPLE_VEC(long, int, uint) \
    __SPIRV_UPSAMPLE_VEC(ulong, uint, uint) \

__SPIRV_UPSAMPLE_TYPES()

#undef __SPIRV_UPSAMPLE_TYPES
#undef __SPIRV_UPSAMPLE_DECL
#undef __SPIRV_UPSAMPLE_VEC
