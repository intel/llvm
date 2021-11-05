//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#if __clang_major__ >= 8
#define CONST_AS __constant
#elif __clang_major__ >= 7
#define CONST_AS __attribute__((address_space(4)))
#else
#define CONST_AS __attribute__((address_space(2)))
#endif

// TODO: implement proper support for global offsets, this also requires
// changes in the compiler and the HIP plugin.
_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalOffset_x() { return 0; }

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalOffset_y() { return 0; }

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalOffset_z() { return 0; }
