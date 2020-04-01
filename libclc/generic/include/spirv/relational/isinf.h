//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define _SPIRV_ISINF_DECL(RET_TYPE, ARG_TYPE) \
  _CLC_OVERLOAD _CLC_DECL RET_TYPE __spirv_IsInf(ARG_TYPE);

#define _SPIRV_VECTOR_ISINF_DECL(RET_TYPE, ARG_TYPE) \
  _SPIRV_ISINF_DECL(RET_TYPE##2, ARG_TYPE##2) \
  _SPIRV_ISINF_DECL(RET_TYPE##3, ARG_TYPE##3) \
  _SPIRV_ISINF_DECL(RET_TYPE##4, ARG_TYPE##4) \
  _SPIRV_ISINF_DECL(RET_TYPE##8, ARG_TYPE##8) \
  _SPIRV_ISINF_DECL(RET_TYPE##16, ARG_TYPE##16)

_SPIRV_ISINF_DECL(int, float)
_SPIRV_VECTOR_ISINF_DECL(int, float)

#ifdef cl_khr_fp64
_SPIRV_ISINF_DECL(int, double)
_SPIRV_VECTOR_ISINF_DECL(long, double)
#endif

#ifdef cl_khr_fp16
_SPIRV_ISINF_DECL(int, half)
_SPIRV_VECTOR_ISINF_DECL(short, half)
#endif

#undef _SPIRV_ISINF_DECL
#undef _SPIRV_VECTOR_ISINF_DECL
