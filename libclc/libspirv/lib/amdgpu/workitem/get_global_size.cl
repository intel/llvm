//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#if __clang_major__ >= 8
#define CONST_AS __constant
#elif __clang_major__ >= 7
#define CONST_AS __attribute__((address_space(4)))
#else
#define CONST_AS __attribute__((address_space(2)))
#endif

#if __clang_major__ >= 6
#define __dispatch_ptr __builtin_amdgcn_dispatch_ptr
#else
#define __dispatch_ptr __clc_amdgcn_dispatch_ptr
CONST_AS uchar *
__clc_amdgcn_dispatch_ptr(void) __asm("llvm.amdgcn.dispatch.ptr");
#endif

_CLC_DEF _CLC_OVERLOAD size_t __spirv_BuiltInGlobalSize(int dim) {
  CONST_AS uint *ptr = (CONST_AS uint *)__dispatch_ptr();
  switch (dim) {
  case 0:
    return ptr[3];
  case 1:
    return ptr[4];
  case 2:
    return ptr[5];
  default:
    return 1;
  }
}
