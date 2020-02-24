//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define _SPIRV_SHUFFLE_DECL(TYPE, MASKTYPE, RETTYPE) \
  _CLC_OVERLOAD _CLC_DECL RETTYPE __spirv_ocl_shuffle(TYPE x, MASKTYPE mask);

//Return type is same base type as the input type, with the same vector size as the mask.
//Elements in the mask must be the same size (number of bits) as the input value.
//E.g. char8 ret = __spirv_ocl_shuffle(char2 x, uchar8 mask);

#define _SPIRV_VECTOR_SHUFFLE_MASKSIZE(INBASE, INTYPE, MASKTYPE) \
  _SPIRV_SHUFFLE_DECL(INTYPE, MASKTYPE##2, INBASE##2) \
  _SPIRV_SHUFFLE_DECL(INTYPE, MASKTYPE##4, INBASE##4) \
  _SPIRV_SHUFFLE_DECL(INTYPE, MASKTYPE##8, INBASE##8) \
  _SPIRV_SHUFFLE_DECL(INTYPE, MASKTYPE##16, INBASE##16) \

#define _SPIRV_VECTOR_SHUFFLE_INSIZE(TYPE, MASKTYPE) \
  _SPIRV_VECTOR_SHUFFLE_MASKSIZE(TYPE, TYPE##2, MASKTYPE) \
  _SPIRV_VECTOR_SHUFFLE_MASKSIZE(TYPE, TYPE##4, MASKTYPE) \
  _SPIRV_VECTOR_SHUFFLE_MASKSIZE(TYPE, TYPE##8, MASKTYPE) \
  _SPIRV_VECTOR_SHUFFLE_MASKSIZE(TYPE, TYPE##16, MASKTYPE) \

_SPIRV_VECTOR_SHUFFLE_INSIZE(char, uchar)
_SPIRV_VECTOR_SHUFFLE_INSIZE(short, ushort)
_SPIRV_VECTOR_SHUFFLE_INSIZE(int, uint)
_SPIRV_VECTOR_SHUFFLE_INSIZE(long, ulong)
_SPIRV_VECTOR_SHUFFLE_INSIZE(uchar, uchar)
_SPIRV_VECTOR_SHUFFLE_INSIZE(ushort, ushort)
_SPIRV_VECTOR_SHUFFLE_INSIZE(uint, uint)
_SPIRV_VECTOR_SHUFFLE_INSIZE(ulong, ulong)
_SPIRV_VECTOR_SHUFFLE_INSIZE(float, uint)
#ifdef cl_khr_fp64
_SPIRV_VECTOR_SHUFFLE_INSIZE(double, ulong)
#endif
#ifdef cl_khr_fp16
_SPIRV_VECTOR_SHUFFLE_INSIZE(half, ushort)
#endif

#undef _SPIRV_SHUFFLE_DECL
#undef _SPIRV_VECTOR_SHUFFLE_MASKSIZE
#undef _SPIRV_VECTOR_SHUFFLE_INSIZE
