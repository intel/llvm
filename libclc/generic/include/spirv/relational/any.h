//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define _SPIRV_ANY_DECL(TYPE) \
  _CLC_OVERLOAD _CLC_DECL int __spirv_Any(TYPE v);

#define _SPIRV_VECTOR_ANY_DECL(TYPE) \
  _SPIRV_ANY_DECL(TYPE)     \
  _SPIRV_ANY_DECL(TYPE##2)  \
  _SPIRV_ANY_DECL(TYPE##3)  \
  _SPIRV_ANY_DECL(TYPE##4)  \
  _SPIRV_ANY_DECL(TYPE##8)  \
  _SPIRV_ANY_DECL(TYPE##16)

_SPIRV_VECTOR_ANY_DECL(char)
_SPIRV_VECTOR_ANY_DECL(short)
_SPIRV_VECTOR_ANY_DECL(int)
_SPIRV_VECTOR_ANY_DECL(long)
