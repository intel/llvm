//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* Duplicate these so we don't have to distribute utils.h */
#define __SPIRV_CONCAT(x, y) x ## y
#define __SPIRV_XCONCAT(x, y) __SPIRV_CONCAT(x, y)

#define __SPIRV_BODY <spirv/relational/select.inc>
#include <spirv/math/gentype.inc>
#define __SPIRV_BODY <spirv/relational/select.inc>
#include <spirv/integer/gentype.inc>

#undef __SPIRV_CONCAT
#undef __SPIRV_XCONCAT
