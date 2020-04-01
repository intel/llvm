//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define _SPIRV_ISEQUAL_DECL(TYPE, RETTYPE) \
  _CLC_OVERLOAD _CLC_DECL RETTYPE __spirv_FOrdEqual(TYPE x, TYPE y);

#define _SPIRV_VECTOR_ISEQUAL_DECL(TYPE, RETTYPE) \
  _SPIRV_ISEQUAL_DECL(TYPE##2, RETTYPE##2)  \
  _SPIRV_ISEQUAL_DECL(TYPE##3, RETTYPE##3)  \
  _SPIRV_ISEQUAL_DECL(TYPE##4, RETTYPE##4)  \
  _SPIRV_ISEQUAL_DECL(TYPE##8, RETTYPE##8)  \
  _SPIRV_ISEQUAL_DECL(TYPE##16, RETTYPE##16)

_SPIRV_ISEQUAL_DECL(float, int)
_SPIRV_VECTOR_ISEQUAL_DECL(float, int)

#ifdef cl_khr_fp64
_SPIRV_ISEQUAL_DECL(double, int)
_SPIRV_VECTOR_ISEQUAL_DECL(double, long)
#endif
#ifdef cl_khr_fp16
_SPIRV_ISEQUAL_DECL(half, int)
_SPIRV_VECTOR_ISEQUAL_DECL(half, short)
#endif

#undef _SPIRV_ISEQUAL_DECL
#undef _SPIRV_VECTOR_ISEQUAL_DEC
