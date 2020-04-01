//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __SPIRV_SCALAR
#define __SPIRV_VECSIZE
#endif

#if __SPIRV_FPSIZE == 64
#define __SPIRV_NATN __SPIRV_XCONCAT(ulong, __SPIRV_VECSIZE)
#elif __SPIRV_FPSIZE == 32
#define __SPIRV_NATN __SPIRV_XCONCAT(uint, __SPIRV_VECSIZE)
#elif __SPIRV_FPSIZE == 16
#define __SPIRV_NATN __SPIRV_XCONCAT(ushort, __SPIRV_VECSIZE)
#endif

_CLC_OVERLOAD _CLC_DECL __SPIRV_GENTYPE __spirv_ocl_nan(__SPIRV_NATN code);

#undef __SPIRV_NATN
#ifdef __SPIRV_SCALAR
#undef __SPIRV_VECSIZE
#endif
