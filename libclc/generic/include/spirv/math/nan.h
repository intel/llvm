//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __SPIRV_CONCAT(x, y) x ## y
#define __SPIRV_XCONCAT(x, y) __SPIRV_CONCAT(x, y)

#define __SPIRV_BODY <spirv/math/nan.inc>
#include <spirv/math/gentype.inc>

#undef __SPIRV_XCONCAT
#undef __SPIRV_CONCAT
