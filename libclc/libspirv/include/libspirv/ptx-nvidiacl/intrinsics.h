//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PTX_NVIDIACL_INTRINSICS_H
#define PTX_NVIDIACL_INTRINSICS_H

_CLC_OVERLOAD long __clc_nvvm_mulhi(long x, long y) __asm("llvm.nvvm.mulhi.ll");
_CLC_OVERLOAD ulong __clc_nvvm_mulhi(ulong x,
                                     ulong y) __asm("llvm.nvvm.mulhi.ull");

#endif
