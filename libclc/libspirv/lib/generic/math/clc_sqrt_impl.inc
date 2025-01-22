//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 64
#define __CLC_NAN __builtin_nan("")
#define ZERO 0.0
#elif __CLC_FPSIZE == 32
#define __CLC_NAN NAN
#define ZERO 0.0f
#elif __CLC_FPSIZE == 16
#define __CLC_NAN (half) NAN
#define ZERO 0.0h
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_sqrt(__CLC_GENTYPE val) {
  return val < ZERO ? __CLC_NAN : __clc_llvm_intr_sqrt(val);
}

#undef __CLC_NAN
#undef ZERO
