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
#define __SPIRV_S_GENTYPE __SPIRV_XCONCAT(long, __SPIRV_VECSIZE)
#define __SPIRV_U_GENTYPE __SPIRV_XCONCAT(ulong, __SPIRV_VECSIZE)
#elif __SPIRV_FPSIZE == 32
#define __SPIRV_S_GENTYPE __SPIRV_XCONCAT(int, __SPIRV_VECSIZE)
#define __SPIRV_U_GENTYPE __SPIRV_XCONCAT(uint, __SPIRV_VECSIZE)
#elif __SPIRV_FPSIZE == 16
#define __SPIRV_S_GENTYPE __SPIRV_XCONCAT(short, __SPIRV_VECSIZE)
#define __SPIRV_U_GENTYPE __SPIRV_XCONCAT(ushort, __SPIRV_VECSIZE)
#endif

_CLC_OVERLOAD _CLC_DECL __SPIRV_GENTYPE __spirv_ocl_select(__SPIRV_GENTYPE x, __SPIRV_GENTYPE y, __SPIRV_S_GENTYPE z);
_CLC_OVERLOAD _CLC_DECL __SPIRV_GENTYPE __spirv_ocl_select(__SPIRV_GENTYPE x, __SPIRV_GENTYPE y, __SPIRV_U_GENTYPE z);

#ifdef __SPIRV_FPSIZE
#undef __SPIRV_S_GENTYPE
#undef __SPIRV_U_GENTYPE
#endif
#ifdef __SPIRV_SCALAR
#undef __SPIRV_VECSIZE
#endif
