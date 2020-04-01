//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DECL __SPIRV_GENTYPE __spirv_ocl_smoothstep(__SPIRV_GENTYPE edge0, __SPIRV_GENTYPE edge1, __SPIRV_GENTYPE x);
_CLC_OVERLOAD _CLC_DECL __SPIRV_GENTYPE __spirv_ocl_smoothstep(float edge0, float edge1, __SPIRV_GENTYPE x);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __SPIRV_GENTYPE __spirv_ocl_smoothstep(double edge0, double edge1, __SPIRV_GENTYPE x);
#endif
