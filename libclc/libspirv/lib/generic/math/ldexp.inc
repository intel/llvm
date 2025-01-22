//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: Enable half precision when ldexp is implemented.
#if __CLC_FPSIZE > 16

#ifndef __CLC_SCALAR

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_ldexp(__CLC_GENTYPE x, int n) {
  return __spirv_ocl_ldexp(x, (__CLC_INTN)n);
}

#endif

#endif
