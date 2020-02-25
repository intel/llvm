//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DECL __SPIRV_GENTYPE __spirv_ocl_mix(__SPIRV_GENTYPE a, __SPIRV_GENTYPE b, __SPIRV_GENTYPE c);

#ifndef __SPIRV_SCALAR
_CLC_OVERLOAD _CLC_DECL __SPIRV_GENTYPE __spirv_ocl_mix(__SPIRV_GENTYPE a, __SPIRV_GENTYPE b, __SPIRV_SCALAR_GENTYPE c);
#endif
