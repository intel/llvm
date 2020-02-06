//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_CONCAT(a, b, c, d, e, f, g) a ## b ## c ## d ## e ## f ## g
#define __CLC_XCONCAT(a, b, c, d, e, f, g) __CLC_CONCAT(a, b, c, d, e, f, g)


#define __SPIRV_DST_ADDR_SPACE local
#define __SPIRV_DST_ADDR_SPACE_MANGLED AS3
#define __SPIRV_SRC_ADDR_SPACE global
#define __SPIRV_SRC_ADDR_SPACE_MANGLED AS1
#define __SPIRV_BODY <spirv/async/async_work_group_strided_copy.inc>
#include <spirv/async/gentype.inc>
#undef __SPIRV_DST_ADDR_SPACE
#undef __SPIRV_DST_ADDR_SPACE_MANGLED
#undef __SPIRV_SRC_ADDR_SPACE
#undef __SPIRV_SRC_ADDR_SPACE_MANGLED
#undef __SPIRV_BODY

#define __SPIRV_DST_ADDR_SPACE global
#define __SPIRV_DST_ADDR_SPACE_MANGLED AS1
#define __SPIRV_SRC_ADDR_SPACE local
#define __SPIRV_SRC_ADDR_SPACE_MANGLED AS3
#define __SPIRV_BODY <spirv/async/async_work_group_strided_copy.inc>
#include <spirv/async/gentype.inc>
#undef __SPIRV_DST_ADDR_SPACE
#undef __SPIRV_DST_ADDR_SPACE_MANGLED
#undef __SPIRV_SRC_ADDR_SPACE
#undef __SPIRV_SRC_ADDR_SPACE_MANGLED
#undef __SPIRV_BODY

#undef __CLC_XCONCAT
#undef __CLC_CONCAT
