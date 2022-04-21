//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/async/common.h>
#include <spirv/spirv.h>

#define __CLC_BODY <async_work_group_strided_copy.inc>
#define __CLC_GEN_VEC3
#include <clc/async/gentype.inc>
