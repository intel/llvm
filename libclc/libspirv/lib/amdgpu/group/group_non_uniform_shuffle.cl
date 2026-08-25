//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

// The __spirv_GroupNonUniform*Shuffle* builtins are emitted directly by the
// SYCL headers (see sycl/detail/spirv.hpp) for sub-group shuffle, permute and
// scan operations, and are relied upon by oneDPL work-group algorithms. Unlike
// SPIR-V targets, the amdgcn target has no runtime translation for these
// instructions, so they must be provided by the device library.

#define __CLC_BODY "group_non_uniform_shuffle.inc"
#include "clc/integer/gentype.inc"

#define __CLC_BODY "group_non_uniform_shuffle.inc"
#include "clc/math/gentype.inc"
